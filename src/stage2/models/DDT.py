"""Class-conditional DiT with a DDT head and internal-guidance base head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from .model_utils import ConditionEmbedder, GaussianFourierEmbedding, NormAttention, RMSNorm, RoPE, SwiGLUFFN


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DDTEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * hidden_size * mlp_ratio))

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(hidden_size, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, rope):
        # These are the DiT's AdaLN residual scales, not learnable layer-fusion weights.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiTwDDTHeadIG(nn.Module):
    """The paper's Base/XL architecture; returns full and shallow-head predictions.

    Parameter names are preserved so the original EMA checkpoints load strictly.
    The shallow head is trained alongside the full head and supplies internal guidance.
    """

    def __init__(
        self,
        input_size=16,
        in_channels=1024,
        patch_size=(1, 1),
        hidden_size=(1440, 2048),
        depth=(28, 2),
        num_heads=(20, 16),
        mlp_ratio=4.0,
        num_classes=1000,
        cond_arch=None,
        base_model_depth=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enc_hidden_size, dec_hidden_size = hidden_size
        self.num_enc_blocks, self.num_dec_blocks = depth
        self.s_patch_size, self.x_patch_size = patch_size
        enc_num_heads, dec_num_heads = num_heads
        if not 1 <= base_model_depth <= self.num_enc_blocks:
            raise ValueError("base_model_depth must select an encoder block")
        self.base_model_depth = base_model_depth

        self.s_embedder = PatchEmbed(input_size, self.s_patch_size, in_channels, self.enc_hidden_size)
        self.x_embedder = PatchEmbed(input_size, self.x_patch_size, in_channels, dec_hidden_size)
        self.s_projector = nn.Linear(self.enc_hidden_size, dec_hidden_size)

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.t_embedder = GaussianFourierEmbedding(self.enc_hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(self.enc_hidden_size, num_classes, cond_arch.num_c_tokens)
        self.blocks = nn.ModuleList(
            [DDTEncoderBlock(self.enc_hidden_size, enc_num_heads, mlp_ratio) for _ in range(self.num_enc_blocks)]
            + [DDTDecoderBlock(dec_hidden_size, dec_num_heads, mlp_ratio) for _ in range(self.num_dec_blocks)]
        )
        self.final_layer = DDTFinalLayer(dec_hidden_size, self.x_patch_size, in_channels)
        self.enc_rope = RoPE(self.enc_hidden_size // enc_num_heads, self.s_embedder.num_patches, self.num_cond_tokens)
        self.dec_rope = RoPE(dec_hidden_size // dec_num_heads, self.x_embedder.num_patches)
        self.initialize_weights()

        self.base_final_layer = DDTFinalLayer(self.enc_hidden_size, self.s_patch_size, in_channels)
        self._zero_final_layer(self.base_final_layer)

    @staticmethod
    def _zero_final_layer(layer):
        nn.init.constant_(layer.adaln_modulation[-1].weight, 0)
        nn.init.constant_(layer.adaln_modulation[-1].bias, 0)
        nn.init.constant_(layer.linear.weight, 0)
        nn.init.constant_(layer.linear.bias, 0)

    def initialize_weights(self):
        for embedder in (self.x_embedder, self.s_embedder):
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.constant_(embedder.proj.bias, 0)
        nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)
        for block in self.blocks[self.num_enc_blocks:]:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self._zero_final_layer(self.final_layer)

    def unpatchify(self, x, p):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]."""
        h, c = int(x.shape[1] ** 0.5), self.in_channels
        return x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, t, context):
        t_emb_base, t_emb = self.t_embedder(t, return_base_embed=True)
        seq = torch.cat([self.s_embedder(x), t_emb, self.ctx_embedder(context)], dim=1)
        for i in range(self.num_enc_blocks):
            seq = self.blocks[i](seq, self.enc_rope)
            if i + 1 == self.base_model_depth:
                x_base = seq[:, :self.s_embedder.num_patches, :]
        seq = self.s_projector(F.silu(t_emb_base + seq[:, :self.s_embedder.num_patches, :]))
        x = self.x_embedder(x)
        for i in range(self.num_dec_blocks):
            x = self.blocks[self.num_enc_blocks + i](x, seq, self.dec_rope)
        x = self.unpatchify(self.final_layer(x, seq), self.x_patch_size)
        x_base = F.silu(t_emb_base + x_base)
        x_base = self.unpatchify(self.base_final_layer(x_base, x_base), self.s_patch_size)
        return x, x_base
