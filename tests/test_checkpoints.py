"""Regression tests for EMA exports and the parameter-free fusion state."""
import sys
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.inference import load_combine_weights, load_decoder_checkpoint, load_dit_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_ema_export_parses_layers_and_rejects_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decoder.safetensors"
            state = {"decoder_embed.weight": torch.ones(2, 3)}
            save_file(state, str(path), metadata={"layers": "[1, 2, 3]", "epoch": "16"})
            decoder, fusion, metadata = load_decoder_checkpoint(path)
            torch.testing.assert_close(decoder["decoder_embed.weight"], state["decoder_embed.weight"])
            self.assertIsNone(fusion)
            self.assertEqual(metadata["layers"], [1, 2, 3])
            self.assertEqual(metadata["epoch"], 16)
            with self.assertRaises(ValueError):
                load_decoder_checkpoint(path, use_ema=False)

    def test_training_checkpoint_selects_ema_not_optimizer_or_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.pt"
            torch.save({"ema": {"module._orig_mod.weight": torch.ones(2)},
                        "model": {"weight": torch.zeros(2)}, "epoch": 40}, path)
            ema, metadata = load_dit_checkpoint(path)
            raw, _ = load_dit_checkpoint(path, use_ema=False)
            torch.testing.assert_close(ema["weight"], torch.ones(2))
            torch.testing.assert_close(raw["weight"], torch.zeros(2))
            self.assertEqual(metadata["epoch"], 40)

    def test_incompatible_fusion_module_is_rejected(self):
        with self.assertRaises(ValueError):
            load_combine_weights(torch.nn.Linear(2, 2), None)
        load_combine_weights(torch.nn.Identity(), None)


if __name__ == "__main__":
    unittest.main()
