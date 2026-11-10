"""The sandbox-tools gates in .github/workflows/build.yml, run in a throwaway repo.

See design/sandbox-tools-ci-gates.md.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[3]
UTILS = "src/inspect_ai/tool/_sandbox_tools_utils"
INJECTABLE_PATHS = f"{UTILS}/injectable_paths.txt"


def _bash_has_mapfile() -> bool:
    # GitHub's runners have bash 5; macOS's own bash 3.2 has no mapfile.
    if shutil.which("bash") is None:
        return False
    check = subprocess.run(["bash", "-c", "type mapfile"], capture_output=True)
    return check.returncode == 0


pytestmark = pytest.mark.skipif(not _bash_has_mapfile(), reason="needs bash 4+")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        + ["-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _pr_repo(repo: Path, base: dict[str, str], pr: dict[str, str]) -> Path:
    """A repo whose origin/main holds the `base` files and whose HEAD edits `pr`."""
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for files in (base, pr):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "commit", "--allow-empty")
        if files is base:
            _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _run_step(
    repo: Path, job: str, step_id: str, **env: str
) -> tuple[int, dict[str, str]]:
    """Run a build.yml step's script in `repo`; return its exit code and outputs."""
    workflow = yaml.safe_load((REPO / ".github/workflows/build.yml").read_text())
    [step] = [s for s in workflow["jobs"][job]["steps"] if s.get("id") == step_id]
    output = repo.parent / "github_output"
    output.write_text("")
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=repo,
        env={**os.environ, "BASE_REF": "main", "GITHUB_OUTPUT": str(output), **env},
        capture_output=True,
        text=True,
    )
    pairs = (line.split("=", 1) for line in output.read_text().splitlines())
    return result.returncode, {key: value for key, value in pairs}


@pytest.mark.parametrize(
    "changed,injectable",
    [
        (f"{UTILS}/Dockerfile.pyinstaller", "true"),
        ("src/inspect_sandbox_tools/src/inspect_sandbox_tools/server.py", "true"),
        ("src/inspect_sandbox_tools/design/notes.md", "false"),
        (f"{UTILS}/sandbox.py", "false"),
    ],
)
def test_detect_slow_classifies_a_change_with_injectable_paths(
    tmp_path: Path, changed: str, injectable: str
) -> None:
    """CI reads the list the runtime's install-state check reads."""
    base = {INJECTABLE_PATHS: (REPO / INJECTABLE_PATHS).read_text(), changed: "1\n"}
    repo = _pr_repo(tmp_path / "repo", base, {changed: "2\n"})

    assert _run_step(repo, "detect-slow", "injectable") == (
        0,
        {"injectable_src": injectable},
    )


@pytest.mark.parametrize(
    "version,bumped", [("9037", True), ("9036", True), ("9035", False), ("9034", False)]
)
def test_version_gate_accepts_any_increase(
    tmp_path: Path, version: str, bumped: bool
) -> None:
    """An injectable change needs a version above the base's, not exactly N+1."""
    version_path = f"{UTILS}/sandbox_tools_version.txt"
    repo = _pr_repo(
        tmp_path / "repo", {version_path: "9035\n"}, {version_path: f"{version}\n"}
    )

    assert _run_step(
        repo, "check-version-bump", "bump", INJECTABLE_SRC_CHANGED="true"
    ) == ((0, {"correctly_bumped": "true"}) if bumped else (1, {}))
