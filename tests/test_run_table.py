import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


module = load("run_table")
planner = load("plan_reproduction")
MANIFEST = json.loads((ROOT / "reproduction/experiments.json").read_text())


def experiment(experiment_id):
    return next(e for e in MANIFEST["experiments"] if e["id"] == experiment_id)


def arguments(**overrides):
    defaults = dict(assets_root=ROOT / "checkpoints", bindings={}, val_npz=Path("val.npz"),
                    results_dir=Path("results"), num_images=None, batch=32, seed=0, device="cuda:0",
                    decoder_output_normalization=None, work_dir=None, python="python", dry_run=True,
                    latent_stats=None)
    defaults.update(overrides)
    return module.SimpleNamespace(**defaults)


def prepared(experiment_id, **overrides):
    return module.prepare(experiment(experiment_id), MANIFEST, module.decoder_configs(),
                          arguments(**overrides))


class RunTableTest(unittest.TestCase):
    def test_every_decoder_rate_resolves_to_its_own_config(self):
        configs = module.decoder_configs()
        self.assertEqual(sorted(configs), [0., .05, .1, .3, .5, .7, .9, .95])
        self.assertEqual(configs[.95].name, "dinov3-k23-p095.yaml")

    def test_feed_selects_the_recorded_layer_readout(self):
        expected = {"k23": None, "k7": "11,13,15,17,19,21,23", "l11": "11"}
        for feed, layers in expected.items():
            with self.subTest(feed=feed):
                command = prepared(f"recon-p0p95-{feed}")["command"]
                self.assertEqual(module.FEED_LAYERS[feed], layers)
                if layers is None:
                    self.assertNotIn("--layers", command)
                else:
                    self.assertEqual(command[command.index("--layers") + 1], layers)

    def test_generation_rows_are_refused_rather_than_approximated(self):
        plan = prepared("swap-p0p0-k23-gnone")
        self.assertEqual(plan["action"], "skip")
        self.assertIn("latent_stats_k23", plan["reason"])

    def test_official_baseline_rows_are_refused(self):
        self.assertEqual(prepared("recon-official-k23-k23")["action"], "skip")

    def test_unresolved_no_drop_normalization_must_be_chosen_explicitly(self):
        self.assertEqual(prepared("recon-p0p0-k23")["action"], "skip")
        self.assertEqual(prepared("recon-p0p0-k23", decoder_output_normalization="raw")["action"], "run")

    def test_smoke_count_overrides_the_fifty_thousand_image_protocol(self):
        command = prepared("recon-p0p95-k23", num_images=100)["command"]
        self.assertEqual(command[command.index("--num-images") + 1], "100")
        full = prepared("recon-p0p95-k23")["command"]
        self.assertEqual(full[full.index("--num-images") + 1], "50000")

    def test_a_checkpoint_whose_size_contradicts_the_inventory_is_refused(self):
        """Wrong-size files are the failure that produces plausible but wrong numbers."""
        candidate = next(c for c in MANIFEST["assets"]["decoder_k23_p0p95"]["candidates"]
                         if c["path"].endswith(".safetensors"))
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / candidate["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            for size, expected in ((candidate["bytes"], "run"), (candidate["bytes"] - 1, "skip")):
                with self.subTest(size=size):
                    with target.open("wb") as handle:
                        handle.truncate(size)
                    plan = prepared("recon-p0p95-k23", assets_root=Path(root), dry_run=False)
                    self.assertEqual(plan["action"], expected)
                    if expected == "skip":
                        self.assertIn("contradicts the inventory", plan["reason"])

    def test_missing_checkpoint_blocks_a_real_run_but_not_a_dry_run(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(prepared("recon-p0p95-k23", assets_root=Path(empty))["action"], "run")
            plan = prepared("recon-p0p95-k23", assets_root=Path(empty), dry_run=False)
            self.assertEqual(plan["action"], "skip")
            self.assertIn("decoder_k23_p0p95", plan["reason"])

    def test_dry_run_of_the_sweep_emits_one_command_per_runnable_row(self):
        with tempfile.TemporaryDirectory() as results:
            code = module.main(["--table", "tab:drop-sweep", "--dry-run", "--results-dir", results])
        self.assertEqual(code, 0)

    def test_paper_table_numbers_select_the_recorded_labels(self):
        numbering = MANIFEST["paper_tables"]["tables"]
        self.assertEqual(numbering["4"]["labels"], ["tab:drop-sweep"])
        for number, entry in numbering.items():
            with self.subTest(table=number):
                by_number = planner.expand_table_selectors(MANIFEST, [number])
                self.assertEqual(by_number, entry["labels"])
                self.assertEqual(planner.expand_table_selectors(MANIFEST, [f"table:{number}"]), entry["labels"])

    def test_unknown_table_number_is_rejected_but_labels_pass_through(self):
        with self.assertRaises(ValueError):
            planner.expand_table_selectors(MANIFEST, ["9"])
        self.assertEqual(planner.expand_table_selectors(MANIFEST, ["appendix"]), ["appendix"])

    def test_generation_rows_need_explicit_latent_statistics(self):
        self.assertEqual(prepared("xl-d0p0-c0p95-gnone")["action"], "skip")
        plan = prepared("xl-d0p0-c0p95-gnone", latent_stats=Path("stats.pt"))
        self.assertEqual(plan["action"], "run")
        self.assertIn("src/eval_fid_dit.py", plan["command"])

    def test_generation_applies_the_same_no_drop_normalization_gate(self):
        self.assertEqual(prepared("xl-d0p0-c0p0-gnone", latent_stats=Path("stats.pt"))["action"], "skip")
        plan = prepared("xl-d0p0-c0p0-gnone", latent_stats=Path("stats.pt"),
                        decoder_output_normalization="encoder")
        self.assertEqual(plan["action"], "run")
        self.assertIn("stage_1.params.decoder_output_normalization=encoder", plan["command"])

    def test_fusion_is_carried_by_the_generator_not_an_encoder_override(self):
        """Sampling never builds the encoder, so k7 rows differ only in generator and statistics."""
        k23 = prepared("swap-p0p95-k23-gnone", latent_stats=Path("s.pt"))["command"]
        k7 = prepared("swap-p0p95-k7-gnone", bindings={"latent_stats_k7": "s7.pt"})["command"]
        for command in (k23, k7):
            self.assertFalse([t for t in command if "encoder_name" in t or "combine_config" in t])
        self.assertNotEqual(k23[k23.index("--ckpt") + 1], k7[k7.index("--ckpt") + 1])
        decoder = "stage_1.params.stage1_ckpt_path=checkpoints/dinov3-vitl/decoder_k23/p0.95.safetensors"
        self.assertIn(decoder, k23)
        self.assertIn(decoder, k7)

    def test_generation_command_binds_generator_decoder_and_guidance(self):
        cases = {"xl-d0p0-c0p95-gnone": ("p00-ditxl", "1.0"), "xl-d0p9-c0p95-g1p78": ("p09-ditxl", "1.78")}
        for experiment_id, (config_marker, scale) in cases.items():
            with self.subTest(experiment=experiment_id):
                command = prepared(experiment_id, latent_stats=Path("stats.pt"))["command"]
                self.assertIn(config_marker, command[command.index("--config") + 1])
                self.assertEqual(command[command.index("--ig-scale") + 1], scale)
                self.assertEqual(command[command.index("--steps") + 1], "50")
                overrides = [t for t in command if t.startswith("stage_1.params.")]
                self.assertEqual(len(overrides), 2)
                self.assertTrue(any("p0.95" in o for o in overrides))

    def test_rows_sharing_latents_differ_only_in_decoder(self):
        group = "generator_xl_p0p0-guidance-None"
        rows = [e for e in MANIFEST["experiments"]
                if e.get("sampling", {}).get("latent_reuse_group") == group]
        self.assertEqual(len(rows), 8)
        fixed, decoders = set(), set()
        for exp in rows:
            command = module.prepare(exp, MANIFEST, module.decoder_configs(),
                                     arguments(latent_stats=Path("stats.pt"),
                                               decoder_output_normalization="encoder"))["command"]
            pick = lambda flag: command[command.index(flag) + 1]
            fixed.add((pick("--ckpt"), pick("--seed"), pick("--batch"), pick("--ig-scale")))
            decoders.add(next(t for t in command if t.startswith("stage_1.params.stage1_ckpt_path")))
        self.assertEqual(len(fixed), 1, "shared latents require one generator, seed, batch and scale")
        self.assertEqual(len(decoders), 8)

    def test_swap_k23_runs_now_that_its_generator_is_assigned(self):
        plan = prepared("swap-p0p95-k23-gnone", latent_stats=Path("stats.pt"))
        self.assertEqual(plan["action"], "run")
        self.assertEqual(plan["checkpoint_asset"], "generator_swap_k23")
        self.assertIn("ep080", plan["command"][plan["command"].index("--ckpt") + 1])

    def test_swap_k7_uses_the_official_generator_and_its_own_statistics(self):
        plan = prepared("swap-p0p95-k7-gnone", bindings={"latent_stats_k7": "stats-k7.pt"})
        self.assertEqual(plan["action"], "run")
        command = plan["command"]
        self.assertIn("raev2-models/stage2", command[command.index("--ckpt") + 1])
        self.assertIn("stage_1.params.normalization_stat_path=stats-k7.pt", command)

    def test_official_guidance_interval_reaches_guided_swap_rows_only(self):
        guided = prepared("swap-p0p95-k23-g1p78", latent_stats=Path("s.pt"))["command"]
        self.assertEqual(guided[guided.index("--ig-tmin") + 1], "0.1")
        self.assertEqual(guided[guided.index("--ig-tmax") + 1], "1.0")
        self.assertNotIn("--ig-tmin", prepared("swap-p0p95-k23-gnone", latent_stats=Path("s.pt"))["command"])
        self.assertNotIn("--ig-tmin", prepared("xl-d0p0-c0p95-g1p78", latent_stats=Path("s.pt"))["command"])

    def test_selection_without_a_mapped_generator_is_not_runnable(self):
        """The DiT-Base half of the two-scale table has no released generators."""
        with tempfile.TemporaryDirectory() as results:
            self.assertEqual(module.main(["--table", "tab:scale-b", "--results-dir", results]), 1)

    def test_paper_numbering_matches_the_readme_reproduction_table(self):
        numbering = MANIFEST["paper_tables"]["tables"]
        self.assertEqual(sorted(numbering), ["1", "2", "4", "5", "6"])
        self.assertEqual(numbering["5"]["labels"], ["tab:dropdecode"])
        self.assertEqual(numbering["6"]["labels"], ["tab:k23_dit_dec_heatmap"])
        self.assertIn("tab:scale-xl", numbering["2"]["labels"])


if __name__ == "__main__":
    unittest.main()
