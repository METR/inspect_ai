from pathlib import Path
from unittest.mock import patch

from inspect_sandbox_tools._cli.server import _ensure_socket_dir


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
