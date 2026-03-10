# Socket Protection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Protect the sandbox tools Unix socket from agent deletion by placing it in a directory that only the server owner can modify.

**Architecture:** Move the socket from `/tmp/sandbox-tools.sock` into `/tmp/inspect-sandbox-tools/sandbox-tools.sock`. The server creates the parent directory on startup. When running as root, the directory is `root:root 0o755` — the agent can connect to the socket but cannot `unlink()` it. Server log files also move into this directory. The client catches `OSError` when it can't delete a stale socket in a protected directory.

**Tech Stack:** Python stdlib (`os`, `pathlib`, `tempfile`, `socket`)

---

### Task 1: Move socket path into subdirectory

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_util/constants.py:7-13`

**Step 1: Update `_get_socket_path` to use a subdirectory**

Replace the entire contents of `constants.py` with:

```python
import tempfile
from pathlib import Path

PKG_NAME = Path(__file__).parent.parent.stem

_SOCKET_DIR_NAME = "inspect-sandbox-tools"
_SOCKET_FILE_NAME = "sandbox-tools.sock"


def _get_socket_dir() -> Path:
    """Get the directory for the Unix domain socket."""
    return Path(tempfile.gettempdir()) / _SOCKET_DIR_NAME


def _get_socket_path() -> Path:
    """Get the Unix domain socket path for the server."""
    return _get_socket_dir() / _SOCKET_FILE_NAME


SOCKET_DIR = _get_socket_dir()
SOCKET_PATH = _get_socket_path()
```

**Step 2: Run existing tests to verify nothing breaks yet**

Run: `cd src/inspect_sandbox_tools && uv run python -m pytest tests/test_job_preexec.py -v`
Expected: PASS (these tests don't touch the socket)

**Step 3: Commit**

```bash
git add src/inspect_sandbox_tools/src/inspect_sandbox_tools/_util/constants.py
git commit -m "refactor: move socket path into subdirectory"
```

---

### Task 2: Create protected directory in server startup

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/server.py:1-54`

**Step 1: Add directory creation before socket binding**

Replace the contents of `server.py` with:

```python
#!/usr/bin/env python3
import os
import socket
import sys
from pathlib import Path

from aiohttp.web import Application, Request, Response, run_app
from jsonrpcserver import async_dispatch

from inspect_sandbox_tools._util.constants import SOCKET_DIR, SOCKET_PATH
from inspect_sandbox_tools._util.load_tools import load_tools

# When running as a PyInstaller onefile binary, all bundled shared libs are extracted
# under sys._MEIPASS. Ensure the dynamic linker can find them by prepending that
# lib directory to LD_LIBRARY_PATH before launching Chromium.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    meipass_lib = Path(sys._MEIPASS) / "lib"
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = f"{meipass_lib}:{existing_ld}" if existing_ld else str(meipass_lib)
    os.environ["LD_LIBRARY_PATH"] = new_ld


def _ensure_socket_dir() -> None:
    """Create the socket directory with appropriate permissions.

    When running as root, the directory is owned by root with mode 0o755.
    This prevents unprivileged users from deleting the socket file (unlinking
    requires write permission on the parent directory).

    When not running as root, the directory is created with mode 0o755 but
    offers no protection since the agent runs as the same user.
    """
    SOCKET_DIR.mkdir(mode=0o755, exist_ok=True)
    if os.getuid() == 0:
        os.chown(SOCKET_DIR, 0, 0)
        os.chmod(SOCKET_DIR, 0o755)


def main():
    load_tools("inspect_sandbox_tools._remote_tools")

    _ensure_socket_dir()

    # Remove stale socket file
    SOCKET_PATH.unlink(missing_ok=True)

    async def handle_request(request: Request) -> Response:
        return Response(
            text=await async_dispatch(await request.text()),
            content_type="application/json",
        )

    app = Application()
    app.router.add_post("/", handle_request)

    # Set umask to handle dynamic user switching scenarios:
    # The server is created on-demand by the first client call, but subsequent
    # calls may come from different users. We must support all combinations:
    # - root creates socket, non-root clients connect later
    # - non-root creates socket, root connects later
    # - non-root1 creates socket, non-root2 connects later
    # Using umask 0o111 creates socket with 0o666 permissions (rw-rw-rw-)
    # allowing any user to connect regardless of who created the socket
    old_umask = os.umask(0o111)
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(SOCKET_PATH))
    finally:
        os.umask(old_umask)

    run_app(app, sock=sock)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/server.py
git commit -m "feat: create protected directory for socket on server startup"
```

---

### Task 3: Move log files and handle protected socket unlink in client

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/main.py:81-82,149-151`

**Step 1: Update log paths and socket unlink error handling**

In `main.py`, make these changes:

1. Change the log path constants (lines 81-82) from:
```python
_SERVER_STDOUT_LOG = "/tmp/sandbox-tools-server-stdout.log"
_SERVER_STDERR_LOG = "/tmp/sandbox-tools-server-stderr.log"
```
to:
```python
_SERVER_STDOUT_LOG = str(SOCKET_DIR / "server-stdout.log")
_SERVER_STDERR_LOG = str(SOCKET_DIR / "server-stderr.log")
```

2. Add `SOCKET_DIR` to the import on line 16. Change:
```python
from inspect_sandbox_tools._util.constants import SOCKET_PATH
```
to:
```python
from inspect_sandbox_tools._util.constants import SOCKET_DIR, SOCKET_PATH
```

3. In `_ensure_server_is_running()`, before opening log files, ensure the directory exists (lines 89-104). Add before the `process = subprocess.Popen(...)` line:
```python
    SOCKET_DIR.mkdir(mode=0o755, exist_ok=True)
```

4. In `_can_connect_to_socket()` (line 151), change:
```python
        SOCKET_PATH.unlink(missing_ok=True)
```
to:
```python
        try:
            SOCKET_PATH.unlink(missing_ok=True)
        except OSError:
            pass
```

**Step 2: Commit**

```bash
git add src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/main.py
git commit -m "feat: move log files to protected dir, handle unlink errors"
```

---

### Task 4: Update test cleanup helpers

**Files:**
- Modify: `src/inspect_sandbox_tools/tests/conftest.py:24-26`
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/test_rpc_integration.py:12-14`

**Step 1: Update `cleanup_socket` in `tests/conftest.py`**

Change the `cleanup_socket` function (line 24-26) from:
```python
def cleanup_socket() -> None:
    """Remove any existing socket file."""
    SOCKET_PATH.unlink(missing_ok=True)
```
to:
```python
def cleanup_socket() -> None:
    """Remove any existing socket file."""
    try:
        SOCKET_PATH.unlink(missing_ok=True)
    except OSError:
        pass
```

**Step 2: Update `cleanup_socket` in `test_rpc_integration.py`**

Change the `cleanup_socket` function (line 12-14) from:
```python
def cleanup_socket():
    """Remove any existing socket file."""
    SOCKET_PATH.unlink(missing_ok=True)
```
to:
```python
def cleanup_socket():
    """Remove any existing socket file."""
    try:
        SOCKET_PATH.unlink(missing_ok=True)
    except OSError:
        pass
```

**Step 3: Commit**

```bash
git add src/inspect_sandbox_tools/tests/conftest.py src/inspect_sandbox_tools/src/inspect_sandbox_tools/_cli/test_rpc_integration.py
git commit -m "fix: handle protected socket dir in test cleanup"
```

---

### Task 5: Add unit test for `_ensure_socket_dir`

**Files:**
- Create: `src/inspect_sandbox_tools/tests/test_socket_dir.py`

**Step 1: Write the test**

```python
import os
from pathlib import Path
from unittest.mock import patch

from inspect_sandbox_tools._cli.server import _ensure_socket_dir
from inspect_sandbox_tools._util.constants import SOCKET_DIR


def test_ensure_socket_dir_creates_directory(tmp_path: Path):
    """_ensure_socket_dir creates the directory with mode 0o755."""
    test_dir = tmp_path / "inspect-sandbox-tools"
    with patch("inspect_sandbox_tools._cli.server.SOCKET_DIR", test_dir):
        with patch("os.getuid", return_value=1000):
            _ensure_socket_dir()
    assert test_dir.exists()
    assert oct(test_dir.stat().st_mode)[-3:] == "755"


def test_ensure_socket_dir_as_root_sets_ownership(tmp_path: Path):
    """When root, _ensure_socket_dir sets root ownership and 0o755."""
    test_dir = tmp_path / "inspect-sandbox-tools"
    with patch("inspect_sandbox_tools._cli.server.SOCKET_DIR", test_dir):
        with patch("os.getuid", return_value=0):
            with patch("os.chown") as mock_chown:
                with patch("os.chmod") as mock_chmod:
                    _ensure_socket_dir()
                    mock_chown.assert_called_once_with(test_dir, 0, 0)
                    mock_chmod.assert_called_once_with(test_dir, 0o755)


def test_ensure_socket_dir_idempotent(tmp_path: Path):
    """Calling _ensure_socket_dir twice does not raise."""
    test_dir = tmp_path / "inspect-sandbox-tools"
    with patch("inspect_sandbox_tools._cli.server.SOCKET_DIR", test_dir):
        with patch("os.getuid", return_value=1000):
            _ensure_socket_dir()
            _ensure_socket_dir()
    assert test_dir.exists()
```

**Step 2: Run the test**

Run: `cd src/inspect_sandbox_tools && uv run python -m pytest tests/test_socket_dir.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/inspect_sandbox_tools/tests/test_socket_dir.py
git commit -m "test: add unit tests for socket directory creation"
```

---

### Task 6: Final verification

**Step 1: Run all sandbox tools tests**

Run: `cd src/inspect_sandbox_tools && uv run python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Run ruff format and lint**

Run: `uv run ruff format src/inspect_sandbox_tools && uv run ruff check --fix src/inspect_sandbox_tools`
Expected: No issues

**Step 3: Squash into a single commit**

Squash all commits from this plan into one:
```bash
git commit -m "feat: protect sandbox tools socket in subdirectory"
```
