"""Publishing commands exercised through subprocesses and local package fixtures."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts/publishing").mkdir(parents=True)
        self.command = self.root / "scripts/publishing.sh"
        source = ROOT / "scripts/publishing.sh"
        if source.exists():
            shutil.copyfile(source, self.command)
        self.lock = self.root / "scripts/publishing/requirements.txt"
        self.receipt = self.root / "receipt.json"
        self.env = dict(
            os.environ,
            UV_NO_INDEX="true",
            UV_CACHE_DIR=str(self.root / "cache"),
            UV_PYTHON_DOWNLOADS="never",
            TWINE_TEST_RECEIPT=str(self.receipt),
        )
        self.wheel = self.root / "twine-1.0-py3-none-any.whl"
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            wheel.writestr("twine/__init__.py", "")
            wheel.writestr(
                "twine/__main__.py",
                "import json, os, pathlib, sys\n"
                "pathlib.Path(os.environ['TWINE_TEST_RECEIPT']).write_text(json.dumps(sys.argv[1:]))\n",
            )
            wheel.writestr("twine-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: twine\nVersion: 1.0\n")
            wheel.writestr(
                "twine-1.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            wheel.writestr("twine-1.0.dist-info/RECORD", "")
        self.write_lock(self.wheel)

    def write_lock(self, artifact):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.lock.write_text(f"twine @ {artifact.as_uri()} --hash=sha256:{digest}\n")

    def run_command(self, *args):
        return subprocess.run(
            ["bash", str(self.command), *args], cwd=self.root, env=self.env, capture_output=True, text=True, check=False
        )

    def test_prepared_environment_forwards_artifact_arguments_without_resolving(self):
        result = self.run_command("sync")
        assert result.returncode == 0, result.stderr
        self.lock.unlink()  # check/upload must not re-resolve or reinstall tools
        for action in ["check", "upload"]:
            result = self.run_command(action, "dist/file with spaces.whl", "dist/file.tar.gz")
            assert result.returncode == 0, result.stderr
            assert json.loads(self.receipt.read_text()) == [action, "dist/file with spaces.whl", "dist/file.tar.gz"]

    def test_tampered_wheel_is_rejected(self):
        with zipfile.ZipFile(self.wheel, "a") as wheel:
            wheel.comment = b"tampered"
        result = self.run_command("sync")
        assert result.returncode != 0
        assert "hash" in result.stderr.lower()
        assert not self.receipt.exists()

    def test_source_only_dependency_never_executes_build_hook(self):
        source = self.root / "twine-1.0.tar.gz"
        marker = self.root / "built"
        payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n".encode()
        with tarfile.open(source, "w:gz") as archive:
            info = tarfile.TarInfo("twine-1.0/setup.py")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        self.write_lock(source)
        result = self.run_command("sync")
        assert result.returncode != 0
        assert "source" in result.stderr.lower()
        assert not marker.exists()

    def test_missing_environment_fails_without_installing_or_uploading(self):
        result = self.run_command("upload", "example.whl")
        assert result.returncode != 0
        assert "sync" in result.stderr
        assert not self.receipt.exists()

    def test_unhashed_requirement_fails(self):
        self.lock.write_text(f"twine @ {self.wheel.as_uri()}\n")
        result = self.run_command("sync")
        assert result.returncode != 0
        assert "hash" in result.stderr.lower()

    def test_incomplete_lock_fails_before_validation_or_upload(self):
        with zipfile.ZipFile(self.wheel) as wheel:
            contents = {name: wheel.read(name) for name in wheel.namelist()}
        contents["twine-1.0.dist-info/METADATA"] += b"Requires-Dist: missing-transitive>=1\n"
        with zipfile.ZipFile(self.wheel, "w") as wheel:
            for name, data in contents.items():
                wheel.writestr(name, data)
        self.write_lock(self.wheel)
        result = self.run_command("sync")
        assert result.returncode != 0
        assert "missing-transitive" in result.stderr
        assert not self.receipt.exists()


class DistributionValidationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PUBLISHING_REAL_PYTHON"), "prepare the real publishing environment")
    def test_real_twine_accepts_wheel_and_sdist_and_rejects_bad_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = (
                "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n"
                "Summary: A fixture\nDescription-Content-Type: text/plain\n\nDescription\n"
            )
            wheel = root / "example-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("example-1.0.dist-info/METADATA", metadata)
                archive.writestr("example-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
            sdist = root / "example-1.0.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                root_entry = tarfile.TarInfo("example-1.0")
                root_entry.type = tarfile.DIRTYPE
                archive.addfile(root_entry)
                payload = metadata.encode()
                info = tarfile.TarInfo("example-1.0/PKG-INFO")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            command = [os.environ["PUBLISHING_REAL_PYTHON"], "-m", "twine", "check"]
            result = subprocess.run([*command, str(wheel), str(sdist)], capture_output=True, text=True, check=False)
            assert result.returncode == 0, result.stdout + result.stderr
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("example-1.0.dist-info/METADATA", "invalid\n")
            result = subprocess.run([*command, str(wheel)], capture_output=True, text=True, check=False)
            assert result.returncode != 0
