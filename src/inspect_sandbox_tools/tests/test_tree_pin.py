"""A running tools process takes nothing from a tree put in place of its own."""

import importlib
import io
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from inspect_sandbox_tools._cli import main as main_module
from inspect_sandbox_tools._util import tree_pin


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A tools tree on sys.path, pinned as a frozen process pins its own."""
    tree = tmp_path / "tree"
    tree.mkdir(mode=0o700)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(tree_pin, "_tree", None)
    monkeypatch.syspath_prepend(str(tree))
    tree_pin.pin_tree(str(tree))
    yield tree
    for name in ("pinned_module", "planted_module"):
        sys.modules.pop(name, None)


def _replace(tree: Path, files: dict[str, str]) -> None:
    """Delete the tree and create another at its path, as a sandbox user could."""
    shutil.rmtree(tree)
    tree.mkdir(mode=0o700)
    for name, text in files.items():
        (tree / name).write_text(text)
    importlib.invalidate_caches()


def test_imports_from_the_tree_work_until_it_is_replaced(
    tree: Path, tmp_path: Path
) -> None:
    ran = tmp_path / "ran"
    (tree / "pinned_module.py").write_text(f"open({str(ran)!r}, 'a').write('pinned')")
    importlib.invalidate_caches()
    importlib.import_module("pinned_module")

    _replace(tree, {"planted_module.py": f"open({str(ran)!r}, 'a').write('planted')"})

    with pytest.raises(ImportError, match="tools directory was replaced"):
        importlib.import_module("planted_module")
    assert ran.read_text() == "pinned"


def test_a_tree_opened_to_other_users_counts_as_replaced(tree: Path) -> None:
    assert not tree_pin.tree_replaced()

    tree.chmod(0o755)

    assert tree_pin.tree_replaced()


def test_a_deleted_tree_counts_as_replaced(tree: Path) -> None:
    shutil.rmtree(tree)

    assert tree_pin.tree_replaced()


def test_a_tree_not_private_at_start_is_not_taken_for_a_replaced_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a sandbox, such as a build's smoke test, the tree need not be 0700."""
    tree = tmp_path / "dist"
    tree.mkdir(mode=0o755)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(tree_pin, "_tree", None)
    tree_pin.pin_tree(str(tree))

    assert not tree_pin.tree_replaced()
    importlib.import_module("colorsys")  # any module not imported yet


@pytest.fixture
def server_dir_in_tree(tree: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The server's files inside the tree, as for a frozen bundle."""
    server_dir = tree / ".server"
    paths = {
        "SERVER_DIR": server_dir,
        "SOCKET_PATH": server_dir / "sandbox-tools.sock",
        "SERVER_PID_PATH": server_dir / "server.pid",
        "SHUTDOWN_STATUS_PATH": server_dir / "shutdown-status.json",
        "_SERVER_START_LOCK_PATH": server_dir / "server-start.lock",
        "_SERVER_STDOUT_LOG": server_dir / "server-stdout.log",
        "_SERVER_STDERR_LOG": server_dir / "server-stderr.log",
    }
    for name, path in paths.items():
        monkeypatch.setattr(main_module, name, path)
    importlib.import_module("psutil")  # as in a bridge that started a server before
    return server_dir


def test_nothing_in_a_replaced_tree_is_touched_to_start_a_server(
    tree: Path, server_dir_in_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", started.append)
    _replace(tree, {})

    with pytest.raises(RuntimeError, match="tools directory was replaced"):
        main_module._ensure_server_is_running()
    assert started == []
    assert list(tree.iterdir()) == []


def test_a_tree_replaced_while_waiting_to_start_the_server_is_refused(
    tree: Path, server_dir_in_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Waiting for another starter can take seconds, so it is checked again."""
    started: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", started.append)
    main_module._ensure_private_server_dirs()
    _replace(tree, {})
    main_module._ensure_private_server_dirs()  # an earlier starter made them again

    with pytest.raises(RuntimeError, match="tools directory was replaced"):
        main_module._ensure_server_is_running_locked()
    assert started == []


def test_the_bridge_ends_once_its_tree_is_replaced(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before taking another request: the host resends it over a checked launch."""

    class Exited(Exception):
        pass

    def exit_process(status: int) -> None:
        raise Exited(status)

    monkeypatch.setattr(os, "_exit", exit_process)
    request = b'{"jsonrpc": "2.0", "method": "remote_version", "id": 1}\n'
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(request)))
    stdout = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(stdout))
    _replace(tree, {})

    with pytest.raises(Exited) as exited:
        main_module._bridge(65536)

    assert exited.value.args == (1,)
    assert stdout.getvalue().splitlines() == [main_module._BRIDGE_READY.encode()]
