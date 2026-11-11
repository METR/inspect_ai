"""Hold a running tools process to the install tree it was launched from.

A frozen bundle keeps reading its tree by path after it starts: modules it has not
imported yet, its extension modules, and the server it starts. Each launch from
the host is checked, but a long-lived process (the server, a bridge, the model
proxy) outlives that check. So it keeps the directory it started from open, which
also stops its inode number being reused, and before each such read checks that
the path still names that directory, with the owner and mode it had at start.
"""

import os
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType

_tree: tuple[str, os.stat_result] | None = None


def pin_tree(path: str) -> None:
    """Pin the tree at `path`, and refuse imports once the path names another one."""
    global _tree
    # Kept open for the life of the process, so no other directory can take the inode.
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    _tree = (path, os.fstat(fd))
    sys.meta_path.insert(0, _TreeGuard())


def tree_replaced() -> bool:
    """Whether the pinned tree's path no longer names it, as it was at start."""
    if _tree is None:
        return False
    path, pinned = _tree
    try:
        now = os.lstat(path)
    except OSError:
        return True
    return _identity(now) != _identity(pinned)


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    """The directory, its owner and its mode (opening it to others counts as a change)."""
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode)


class _TreeGuard:
    """First on `sys.meta_path`: an import not yet done fails once the tree is gone."""

    @staticmethod
    def find_spec(
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if tree_replaced():
            raise ImportError(
                f"Cannot import {fullname}: the sandbox tools directory was replaced"
            )
        return None
