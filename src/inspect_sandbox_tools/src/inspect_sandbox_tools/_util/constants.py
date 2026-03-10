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
