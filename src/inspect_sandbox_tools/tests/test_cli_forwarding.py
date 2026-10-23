"""The ``exec`` CLI forwards server-handled requests using only the standard library."""

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

import pytest
from inspect_sandbox_tools._cli import main as main_module
from inspect_sandbox_tools._cli.jsonrpc_incoming import JSONRPCIncoming
from inspect_sandbox_tools._util.load_tools import load_tools
from pydantic import ValidationError

from tests.conftest import RpcClient

# Most of an `exec` trip's startup cost, before forwarding stopped loading them.
_HEAVY_MODULES = [
    "aiohttp",
    "asyncio",
    "importlib.metadata",
    "jsonrpcserver",
    "psutil",
    "pydantic",
    "inspect_sandbox_tools._agent_bridge.proxy",
    "inspect_sandbox_tools._cli.server",
    "inspect_sandbox_tools._util.load_tools",
]


@pytest.mark.usefixtures("sandbox_server_cleanup")
def test_forwarding_does_not_import_heavy_modules(rpc_client: RpcClient) -> None:
    # Starting the server may load psutil; forwarding to a running one may not.
    rpc_client({"jsonrpc": "2.0", "method": "remote_version", "id": 1}, 10)
    report_imports = (
        "import json, sys\n"
        "from inspect_sandbox_tools._cli.main import main\n"
        "main()\n"
        f"print(json.dumps([m for m in {_HEAVY_MODULES!r} if m in sys.modules]),"
        " file=sys.stderr)\n"
    )
    request = {
        "jsonrpc": "2.0",
        "method": "exec_remote_poll",
        "params": {"pid": 2**31 - 1, "ack_seq": 0},
        "id": 2,
    }

    result = subprocess.run(
        [sys.executable, "-c", report_imports, "exec", json.dumps(request)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["error"]["code"] == -32099  # no such job
    assert json.loads(result.stderr.splitlines()[-1]) == []


def test_in_process_tools_match_the_tool_registry() -> None:
    assert main_module._IN_PROCESS_TOOLS == load_tools(
        "inspect_sandbox_tools._in_process_tools"
    )


@pytest.mark.parametrize(
    "request_data",
    [
        {"jsonrpc": "2.0", "method": "m", "id": 1},
        {"jsonrpc": "2.0", "method": "m", "params": [1], "id": 1},
        {"jsonrpc": "2.0", "method": "m", "params": {"a": 1}},
        {"jsonrpc": "2.0", "method": "m", "params": None, "extra": True},
        {"method": "m"},
        {"jsonrpc": "1.0", "method": "m"},
        {"jsonrpc": 2.0, "method": "m"},
        {"jsonrpc": "2.0"},
        {"jsonrpc": "2.0", "method": 1},
        {"jsonrpc": "2.0", "method": None},
        {"jsonrpc": "2.0", "method": "m", "params": "p"},
        {"jsonrpc": "2.0", "method": "m", "params": True},
        [{"jsonrpc": "2.0", "method": "m"}],
        "m",
        None,
    ],
)
def test_request_method_matches_jsonrpc_incoming(request_data: object) -> None:
    try:
        expected = JSONRPCIncoming.model_validate(request_data).method
    except ValidationError as ex:
        with pytest.raises(ValidationError) as raised:
            main_module._request_method(request_data)
        assert str(raised.value) == str(ex)
    else:
        assert main_module._request_method(request_data) == expected


@pytest.mark.parametrize(
    "reply",
    [b"", b"HTTP/1.0 200 OK\r\nContent-Length: 10\r\n\r\n{}"],
    ids=["no response", "truncated body"],
)
def test_forwarding_fails_on_an_incomplete_response(reply: bytes) -> None:
    directory = tempfile.mkdtemp(dir="/tmp")  # short enough for a socket path
    socket_path = f"{directory}/server.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(socket_path)
            server.listen(1)

            def reply_once() -> None:
                connection, _ = server.accept()
                with connection:
                    connection.recv(65536)
                    connection.sendall(reply)

            thread = threading.Thread(target=reply_once)
            thread.start()
            with pytest.raises(ConnectionError, match="complete response"):
                main_module._json_rpc_unix_call(socket_path, "{}")
            thread.join()
    finally:
        shutil.rmtree(directory)
