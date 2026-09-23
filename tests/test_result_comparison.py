import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("compare_result", Path(__file__).resolve().parents[1] / "scripts/compare_result.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResultComparisonTest(unittest.TestCase):
    experiment = {"id": "example", "num_samples": 50000, "paper_reported": {"gfid": 2., "is": 200.}}

    def test_small_run_never_becomes_formal_reproduction(self):
        result = module.compare(self.experiment, {"num_samples": 100, "fid": 2.5, "is": 190.})
        self.assertEqual(result["status"], "smoke_only")
        self.assertFalse(result["protocol_verified"])
        self.assertEqual(result["metrics"][0]["absolute_difference"], .5)

    def test_equal_numbers_do_not_certify_protocol(self):
        result = module.compare(self.experiment, {"num_samples": 50000, "fid": 2., "is": 200.})
        self.assertEqual(result["status"], "measured_protocol_unverified")
        self.assertFalse(result["protocol_verified"])

    def test_missing_or_nan_measurements_rejected(self):
        for measured in ({"num_samples": 100}, {"num_samples": 100, "fid": float("nan"), "is": 200.}):
            with self.assertRaises(ValueError):
                module.compare(self.experiment, measured)
