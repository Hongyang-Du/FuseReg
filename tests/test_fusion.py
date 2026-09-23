"""Scientific invariants of the released layer fusion, using tiny CPU tensors."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage1.combine import MLSCombine


class FusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.tokens = [torch.randn(4, 5, 3) for _ in range(3)]

    def combine(self, **options):
        return MLSCombine(layers=[1, 2, 3], **options)

    def test_eval_full_feed_ignores_training_dropout(self):
        expected = torch.stack(self.tokens).mean(0) + self.tokens[-1].mean(1, keepdim=True)
        for p in (0, 0.3, 0.95):
            with self.subTest(p=p):
                model = self.combine(p_drop=p).eval()
                torch.testing.assert_close(model(self.tokens), expected)

    def test_subset_retains_fixed_last_layer_surrogate(self):
        model = self.combine().eval()
        expected = self.tokens[0] + self.tokens[-1].mean(1, keepdim=True)
        torch.testing.assert_close(model(self.tokens, idx=[0]), expected)
        # Moving the omitted last layer changes the offset, not subset membership.
        shifted = self.tokens[:-1] + [self.tokens[-1] + 2]
        torch.testing.assert_close(model(shifted, idx=[0]), expected + 2)

    def test_empty_bernoulli_draw_keeps_one_layer(self):
        model = self.combine(p_drop=1.0).train()
        tokens = [torch.full((128, 2, 3), float(i + 1)) for i in range(3)]
        out = model(tokens) - tokens[-1].mean(1, keepdim=True)
        self.assertTrue(torch.isfinite(out).all())
        # If every layer is dropped, each image receives one complete layer.
        self.assertTrue(((out == 1) | (out == 2) | (out == 3)).all())
        torch.testing.assert_close(out, out[:, :1, :1].expand_as(out))

    def test_parameter_free_fusion_and_subset_renormalization(self):
        model = self.combine().eval()
        self.assertFalse(model.has_params)
        torch.testing.assert_close(model(self.tokens, idx=[0, 2]), (self.tokens[0] + self.tokens[2]) / 2 + self.tokens[-1].mean(1, keepdim=True))


if __name__ == "__main__":
    unittest.main()
