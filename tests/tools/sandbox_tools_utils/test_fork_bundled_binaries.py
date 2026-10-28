"""METR fork: the committed sandbox-tools binaries match SHA256SUMS and their source.

Hawk installs this fork from git, so the binaries ship in `src/inspect_ai/binaries`
instead of coming from upstream's bucket. When the sandbox-tools source or its
build inputs change, rebuild all four
(`build_within_container.py --arch {amd64,arm64} [--musl] --dev=false`), rewrite
`SHA256SUMS`, and update `SOURCE_SHA256` by running this file with Python.
"""

import base64
import hashlib
import re
import tarfile
from pathlib import Path

import pytest

import inspect_ai
from inspect_ai.tool._sandbox_tools_utils._build_config import (
    SandboxToolsBuildConfig,
    config_to_filename,
)
from inspect_ai.tool._sandbox_tools_utils._digests import read_sha256sums
from inspect_ai.tool._sandbox_tools_utils.sandbox import _get_sandbox_tools_version

SRC = Path(inspect_ai.__file__).parents[1]
UTILS = SRC / "inspect_ai" / "tool" / "_sandbox_tools_utils"
SOURCE_SHA256 = UTILS / "SOURCE_SHA256"
BUILD_INPUTS = [
    "Dockerfile.pyinstaller",
    "Dockerfile.pyinstaller.musl",
    "_build_bundled_executable.py",
    "_build_config.py",
    "build_executable.py",
    "build_within_container.py",
    "sandbox_tools_version.txt",
]
PACKAGE = SRC / "inspect_sandbox_tools" / "src" / "inspect_sandbox_tools"

pytestmark = pytest.mark.skipif(
    not PACKAGE.is_dir(), reason="needs the source checkout, not an installed package"
)


def _package_files() -> list[Path]:
    """The package's source files (no bytecode caches or dot files)."""
    return [
        f
        for f in PACKAGE.rglob("*")
        if f.is_file()
        and not any(
            part == "__pycache__" or part.startswith(".")
            for part in f.relative_to(PACKAGE).parts
        )
    ]


def _artifacts() -> list[Path]:
    version = int(_get_sandbox_tools_version())
    return [
        SRC
        / "inspect_ai"
        / "binaries"
        / config_to_filename(
            SandboxToolsBuildConfig(arch=arch, version=version, suffix=None, musl=musl)
        )
        for arch in ("amd64", "arm64")
        for musl in (False, True)
    ]


def source_sha256() -> str:
    """Digest of the sandbox-tools package source and the inputs that build it."""
    files = _package_files()
    files += [SRC / "inspect_sandbox_tools" / "pyproject.toml"]
    files += [UTILS / name for name in BUILD_INPUTS]
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.relative_to(SRC).as_posix()):
        entry = f"{f.relative_to(SRC).as_posix()}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n"
        digest.update(entry.encode())
    return digest.hexdigest()


def test_bundled_binaries_match_sha256sums() -> None:
    sums = read_sha256sums()
    for binary in _artifacts():
        assert binary.is_file(), f"{binary.name} is not committed"
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == sums[binary.name]


def test_bundled_binaries_contain_this_source() -> None:
    """Each binary's bundled package RECORD matches the package source, file by file."""
    expected = {
        f"inspect_sandbox_tools/{f.relative_to(PACKAGE).as_posix()}": _record_digest(f)
        for f in _package_files()
    }
    record_name = re.compile(r"_internal/inspect_sandbox_tools-[^/]+\.dist-info/RECORD")
    for binary in _artifacts():
        with tarfile.open(binary, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if record_name.fullmatch(m.name))
            extracted = tar.extractfile(member)
            assert extracted is not None
            record = extracted.read().decode()
        bundled = {}
        for line in record.splitlines():
            name, digest, _ = line.rsplit(",", 2)
            if name.startswith("inspect_sandbox_tools/") and digest.startswith(
                "sha256="
            ):
                bundled[name] = digest.removeprefix("sha256=")
        assert bundled == expected, binary.name


def _record_digest(f: Path) -> str:
    """A file's digest as a wheel RECORD writes it (urlsafe base64, unpadded)."""
    digest = hashlib.sha256(f.read_bytes()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def test_bundled_binaries_were_built_from_this_source() -> None:
    assert source_sha256() == SOURCE_SHA256.read_text().strip(), (
        "the sandbox-tools source or its build inputs changed since the binaries "
        "were built: rebuild all four, rewrite SHA256SUMS, then run this file to "
        "update SOURCE_SHA256"
    )


if __name__ == "__main__":
    SOURCE_SHA256.write_text(source_sha256() + "\n")
    print(f"wrote {SOURCE_SHA256}")
