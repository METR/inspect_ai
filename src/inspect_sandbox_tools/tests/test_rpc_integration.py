"""End-to-end checks of the ``exec`` CLI entry point against a real server."""

import base64
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from inspect_sandbox_tools._util.constants import SERVER_PID_PATH, SOCKET_PATH

from tests.conftest import RpcClient

pytestmark = pytest.mark.usefixtures("sandbox_server_cleanup")


def _run_exec(stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "inspect_sandbox_tools._cli.main", "exec"],
        input=stdin,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_version_method(rpc_client: RpcClient) -> None:
    """The version method is answered in-process."""
    response = rpc_client({"jsonrpc": "2.0", "method": "version", "id": 1}, 10)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "error" not in response
    version = response["result"]
    assert isinstance(version, str)
    assert len(version.split(".")) == 3


def test_socket_creation_and_permissions(rpc_client: RpcClient) -> None:
    """The first remote call starts the server with state private to its user."""
    rpc_client({"jsonrpc": "2.0", "method": "version", "id": 1}, 10)
    assert not SOCKET_PATH.exists(), "Socket should not exist after in-process call"

    response: dict[str, Any] = rpc_client(
        {"jsonrpc": "2.0", "method": "bash_session_new_session", "id": 666}, 10
    )
    assert response["result"]["session_name"] == "BashSession"

    for _ in range(50):
        if SOCKET_PATH.exists():
            break
        time.sleep(0.1)
    else:
        pytest.fail("Socket was not created within 5 seconds")

    # Only the CLI wrapper, running as the server's user, may connect.
    socket_mode = SOCKET_PATH.stat().st_mode & 0o777
    assert socket_mode & 0o077 == 0, (
        f"Socket is reachable by other users: {socket_mode:o}"
    )
    directory_mode = SOCKET_PATH.parent.stat().st_mode & 0o777
    assert directory_mode == 0o700, (
        f"Server directory should be 0700, got {directory_mode:o}"
    )


def test_invalid_json_request() -> None:
    result = _run_exec("invalid json")

    assert result.returncode != 0
    assert "error" in result.stderr.lower()


def test_malformed_jsonrpc_request() -> None:
    result = _run_exec(json.dumps({"method": "version"}))

    assert result.returncode != 0


class _Bridge:
    """A `bridge` CLI process: requests in on stdin, replies out on stdout."""

    def __init__(self, max_frame_bytes: int = 65536) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "inspect_sandbox_tools._cli.main",
                "bridge",
                "--max-frame-bytes",
                str(max_frame_bytes),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert json.loads(self.read_line()) == {
            "jsonrpc": "2.0",
            "method": "__inspect_bridge_ready__",
            "params": {"version": 1},
        }

    def send(self, request: dict[str, Any]) -> None:
        assert self.process.stdin
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()

    def read_line(self, timeout: float = 10) -> bytes:
        assert self.process.stdout
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        assert readable, "no reply from the bridge"
        return self.process.stdout.readline()

    def read(self) -> dict[str, Any]:
        return json.loads(self.read_line())

    def read_reply(self, timeout: float) -> dict[str, Any]:
        return json.loads(self.read_line(timeout))

    def close(self) -> int:
        assert self.process.stdin
        self.process.stdin.close()
        return self.process.wait(10)


def _wait_until_refused(socket_path: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            try:
                probe.connect(str(socket_path))
            except OSError:
                return
        time.sleep(0.05)
    pytest.fail(f"{socket_path} still accepts connections")


def _start_wait(command: str, request_id: str, id: int) -> dict[str, Any]:
    params = {"command": command, "request_id": request_id, "wait_seconds": 5.0}
    return {
        "jsonrpc": "2.0",
        "method": "exec_remote_start_wait",
        "params": params,
        "id": id,
    }


def test_bridge_answers_requests_while_another_is_held(tmp_path: Path) -> None:
    release = tmp_path / "release"
    bridge = _Bridge()
    bridge.send(
        _start_wait(f"while [ ! -e {release} ]; do sleep 0.05; done", uuid4().hex, 1)
    )
    bridge.send({"jsonrpc": "2.0", "method": "remote_version", "id": 2})

    # the server holds the first request until its command ends, which needs `release`
    assert bridge.read()["id"] == 2
    release.touch()
    assert bridge.read()["id"] == 1
    assert bridge.close() == 0


def test_bridge_spills_a_reply_longer_than_the_frame_limit() -> None:
    bridge = _Bridge(max_frame_bytes=4096)
    bridge.send(_start_wait("printf 'x%.0s' $(seq 20000)", uuid4().hex, 1))

    line = bridge.read_line()
    assert len(line) <= 4096
    response = b""
    chunk = json.loads(line)["__inspect_json_rpc_response_chunk__"]
    while True:
        response += base64.b64decode(chunk["chunk"])
        if chunk["done"]:
            break
        params = {"handle": chunk["handle"], "offset": chunk["next_offset"]}
        bridge.send(
            {
                "jsonrpc": "2.0",
                "method": "__inspect_json_rpc_response_chunk__",
                "params": params,
                "id": 2,
            }
        )
        line = bridge.read_line()
        assert len(line) <= 4096
        chunk = json.loads(line)["__inspect_json_rpc_response_chunk__"]

    assert json.loads(response)["result"]["result"]["stdout"] == "x" * 20000
    assert bridge.close() == 0


def test_bridge_restarts_a_server_that_died() -> None:
    bridge = _Bridge()
    request_id = uuid4().hex
    bridge.send(_start_wait("true", request_id, 1))
    assert bridge.read()["result"]["result"]["state"] == "completed"
    server_pid = json.loads(SERVER_PID_PATH.read_text())["pid"]
    os.kill(server_pid, signal.SIGKILL)
    _wait_until_refused(SOCKET_PATH)

    bridge.send({"jsonrpc": "2.0", "method": "remote_version", "id": 2})
    assert "result" in bridge.read_reply(timeout=30)
    assert json.loads(SERVER_PID_PATH.read_text())["pid"] != server_pid
    # the new server knows the old one launched it, so a resent start is not rerun
    bridge.send(_start_wait("true", request_id, 3))
    assert "may have run before the sandbox tools server restarted" in str(
        bridge.read()
    )
    assert bridge.close() == 0


@pytest.mark.parametrize(
    "request_data",
    [
        {"jsonrpc": "2.0", "method": "version", "id": 1},
        {
            "jsonrpc": "2.0",
            "method": "exec_remote_start",
            "params": {"command": "id", "_run_as": "nobody"},
            "id": 1,
        },
    ],
    ids=["in-process tool", "run as another user"],
)
def test_bridge_refuses_requests_it_cannot_forward(
    request_data: dict[str, Any],
) -> None:
    bridge = _Bridge()
    bridge.send(request_data)

    reply = bridge.read()
    assert reply["id"] == 1
    assert "cannot be sent over the bridge" in reply["__inspect_bridge_error__"]
    assert bridge.close() == 0


@pytest.mark.parametrize(
    "line",
    [
        b"not json\n",
        b'{"jsonrpc": "2.0", "method": "remote_version"}\n',
        b"x" * 200 + b"\n",
    ],
    ids=["not JSON", "no ID", "longer than the frame limit"],
)
def test_bridge_exits_on_a_line_that_is_not_a_request(line: bytes) -> None:
    bridge = _Bridge(max_frame_bytes=100)
    assert bridge.process.stdin
    bridge.process.stdin.write(line)
    bridge.process.stdin.flush()

    assert bridge.process.wait(10) != 0
