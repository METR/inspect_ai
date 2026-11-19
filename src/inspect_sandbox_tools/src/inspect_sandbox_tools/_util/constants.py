import os
import sys
from pathlib import Path

from inspect_sandbox_tools._util.server_dir import (
    SERVER_DIR_ENV,
    resolve_server_dir,
    server_socket_path,
)

__all__ = [
    "PKG_NAME",
    "SERVER_DIR",
    "EXEC_REMOTE_JOBS_PATH",
    "SERVER_DIR_ENV",
    "SERVER_PID_PATH",
    "SHUTDOWN_STATUS_PATH",
    "SOCKET_PATH",
]

PKG_NAME = Path(__file__).parent.parent.stem

SERVER_DIR = resolve_server_dir(
    os.environ, bool(getattr(sys, "frozen", False)), sys.executable
)
SOCKET_PATH = server_socket_path(SERVER_DIR)
SHUTDOWN_STATUS_PATH = SERVER_DIR / "shutdown-status.json"
SERVER_PID_PATH = SERVER_DIR / "server.pid"
# METR fork: the keys of the exec_remote jobs a server has started (see
# exec_remote's Controller.persist_launches).
EXEC_REMOTE_JOBS_PATH = SERVER_DIR / "exec-remote-jobs"
