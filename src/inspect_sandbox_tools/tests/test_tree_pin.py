"""A running tools process takes nothing from a tree put in place of its own."""

import importlib
import io
import os
import shutil
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


def test_no_server_is_started_from_a_replaced_tree(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []
    monkeypatch.setattr(main_module.subprocess, "Popen", started.append)
    _replace(tree, {})

    with pytest.raises(RuntimeError, match="tools directory was replaced"):
        main_module._ensure_server_is_running()
    assert started == []


def test_a_tree_replaced_while_waiting_to_start_the_server_is_refused(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Waiting for another starter can take seconds, so it is checked again."""
    importlib.import_module("psutil")  # as in a bridge that started the server before
    started: list[object] = []
    monkeypatch.setattr(main_module.subprocess, "Popen", started.append)
    for name in ("SOCKET_PATH", "SERVER_PID_PATH", "SHUTDOWN_STATUS_PATH"):
        monkeypatch.setattr(main_module, name, tmp_path / name)
    for name in ("_SERVER_STDOUT_LOG", "_SERVER_STDERR_LOG"):
        monkeypatch.setattr(main_module, name, tmp_path / name)

    def replaced_meanwhile() -> bool:
        _replace(tree, {})
        return False

    monkeypatch.setattr(main_module, "_can_connect_to_socket", replaced_meanwhile)

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

    taken: list[object] = []
    monkeypatch.setattr(main_module, "_bridge_reply", lambda *args: taken.append(args))
    monkeypatch.setattr(os, "_exit", exit_process)
    request = b'{"jsonrpc": "2.0", "method": "remote_version", "id": 1}\n'
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(request)))
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO()))
    _replace(tree, {})

    with pytest.raises(Exited) as exited:
        main_module._bridge(65536)

    assert exited.value.args == (1,)
    assert taken == []
