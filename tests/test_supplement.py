import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


spec = importlib.util.spec_from_file_location("build_supplement", Path(__file__).resolve().parents[1] / "scripts/build_supplement.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SupplementTest(unittest.TestCase):
    def test_source_allowlist_and_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs/README_ANONYMOUS.md").write_text("Anonymous reproduction instructions\n")
            (root / "README.md").write_text("Author information\n")
            (root / ".env").write_text("credential must never be copied\n")
            (root / "checkpoint.pt").write_bytes(b"checkpoint")
            (root / "src").mkdir()
            (root / "src/main.py").write_text("print('ready')\n")
            (root / "src/generated.egg-info").mkdir()
            (root / "src/generated.egg-info/PKG-INFO").write_text("Build metadata\n")
            (root / "src/generated.egg-info/dependency_links.txt").write_text("Build metadata\n")
            (root / "src/data").mkdir()
            (root / "src/data/__init__.py").write_text("# Source data loaders\n")
            first, second = root / "first.zip", root / "second.zip"
            module.build(root, first)
            module.build(root, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.read("FuseReg/README.md"), b"Anonymous reproduction instructions\n")
                self.assertNotIn("FuseReg/.env", archive.namelist())
                self.assertNotIn("FuseReg/checkpoint.pt", archive.namelist())
                self.assertIn("FuseReg/src/data/__init__.py", archive.namelist())
                self.assertFalse(any("egg-info" in name for name in archive.namelist()))
                for line in archive.read("FuseReg/MANIFEST.sha256").decode().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(digest, hashlib.sha256(archive.read(name)).hexdigest())

    def test_private_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs/README_ANONYMOUS.md").write_text("Anonymous\n")
            (root / "README.md").write_text("README\n")
            (root / "docs/leak.md").write_text("/" + "Users" + "/researcher/private\n")
            with self.assertRaises(ValueError):
                module.build(root, root / "out.zip")


if __name__ == "__main__":
    unittest.main()
