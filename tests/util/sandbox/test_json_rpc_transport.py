import base64
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from functools import partial
from typing import Any, Literal, overload

import anyio
import anyio.lowlevel
import pytest
from anyio.abc import ByteStream, TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from test_helpers.sandbox import CannedSandbox
from typing_extensions import override

from inspect_ai._eval.task.sandbox import sandboxenv_context
from inspect_ai._util._async import tg_collect
from inspect_ai.dataset import Sample
from inspect_ai.event import SandboxEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.util import sandbox as current_sandbox
from inspect_ai.util._sandbox import _json_rpc_bridge as json_rpc_bridge
from inspect_ai.util._sandbox._cli import (
    SANDBOX_CLI,
    TOOLS_DIR_REFUSED,
    tools_cli_command,
)
from inspect_ai.util._sandbox._json_rpc_bridge import BridgeDropped, close_bridges
from inspect_ai.util._sandbox._json_rpc_transport import (
    SandboxJSONRPCTransport,
    SandboxReplacedError,
)
from inspect_ai.util._sandbox.environment import (
    RootAccess,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentSpec,
)
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy, SandboxTimeoutError
from inspect_ai.util._sandbox.exec_remote import (
    ExecRemoteAwaitableOptions,
    exec_remote_awaitable,
)
from inspect_ai.util._sandbox.limits import (
    OutputLimitExceededError,
    override_max_exec_output_size,
    override_max_read_file_size,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from inspect_ai.util._subprocess import ExecResult

_CHUNK_FIELD = "__inspect_json_rpc_response_chunk__"
_CHUNK_METHOD = "__inspect_json_rpc_response_chunk__"


async def test_sandbox_json_rpc_transport_reassembles_large_unicode_response() -> None:
    original_response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "stdout": "stdout-" + "🙂" * 1000,
                "stderr": "stderr-" + "界" * 1000,
            },
        },
        ensure_ascii=False,
    )
    original_bytes = original_response.encode("utf-8")
    exec_inputs: list[dict[str, Any]] = []
    handle = "a" * 32

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            exec_inputs.append(kwargs)
            request = json.loads(kwargs["input"])
            if request["method"] == _CHUNK_METHOD:
                params = request["params"]
                if params.get("release") is True:
                    return ExecResult(True, 0, _success_response(), "")
                return ExecResult(
                    True,
                    0,
                    _chunk_response(original_bytes, handle, params["offset"], 127),
                    "",
                )
            return ExecResult(
                True, 0, _chunk_response(original_bytes, handle, 0, 127), ""
            )

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    with override_max_exec_output_size(1024):
        assert await transport("large", {}, False) == original_response

    continuation_offsets = [
        json.loads(call["input"])["params"]["offset"] for call in exec_inputs[1:-1]
    ]
    assert continuation_offsets == sorted(continuation_offsets)
    assert len(continuation_offsets) > 2
    assert all(
        call["env"]["INSPECT_SANDBOX_JSON_RPC_RESPONSE_MAX_BYTES"] == "1024"
        for call in exec_inputs
    )
    assert json.loads(exec_inputs[-1]["input"])["params"]["release"] is True


async def test_a_chunked_reply_whose_release_is_refused_fails_as_a_replaced_sandbox() -> (
    None
):
    """The reply came in whole, but the sandbox was replaced before its release."""
    reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "x" * 100}).encode()
    offsets: list[int] = []

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        params = json.loads(sandbox.inputs[-1] or "").get("params", {})
        if params.get("release"):
            return ExecResult(True, 0, f"{TOOLS_DIR_REFUSED}\n", "")
        offsets.append(params.get("offset", 0))
        return ExecResult(
            True, 0, _chunk_response(reply, "c" * 32, offsets[-1], 100), ""
        )

    sandbox = CannedSandbox(policy)
    transport = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)
    with override_max_exec_output_size(64), pytest.raises(SandboxReplacedError):
        await transport("big", {}, False)

    assert offsets == [0, 100]  # every chunk was fetched before the release


@pytest.mark.parametrize("over_the_limit", [True, False])
async def test_a_chunked_reply_is_capped_at_the_read_file_limit(
    over_the_limit: bool,
) -> None:
    """Its first chunk declares the size; one over the limit fails only that call."""
    reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "x" * 5000}).encode()
    requests: list[dict[str, Any]] = []

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        requests.append(json.loads(sandbox.inputs[-1] or ""))
        params = requests[-1].get("params", {})
        if params.get("release"):
            return ExecResult(True, 0, _success_response(), "")
        offset = params.get("offset", 0)
        return ExecResult(True, 0, _chunk_response(reply, "c" * 32, offset, 2048), "")

    sandbox = CannedSandbox(policy)
    transport = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)
    with (
        override_max_exec_output_size(4096),
        override_max_read_file_size(len(reply) - over_the_limit),
    ):
        if over_the_limit:
            with pytest.raises(OutputLimitExceededError) as raised:
                await transport("big", {}, False)
            assert raised.value.limit_str == f"{len(reply) - 1} bytes"
        else:
            assert await transport("big", {}, False) == reply.decode()

    fetched = [r["params"].get("offset") for r in requests[1:-1]]
    assert fetched == ([] if over_the_limit else list(range(2048, len(reply), 2048)))
    assert requests[-1]["params"] == {"handle": "c" * 32, "release": True}


async def test_a_refused_release_does_not_replace_an_error_on_its_way() -> None:
    """A cancellation, say: the call fails with what stopped it."""
    reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "x" * 100}).encode()

    def policy(cmd: list[str], user: str | None) -> ExecResult[str]:
        params = json.loads(sandbox.inputs[-1] or "").get("params", {})
        if params.get("release"):
            return ExecResult(True, 0, f"{TOOLS_DIR_REFUSED}\n", "")
        if params.get("offset"):
            return ExecResult(True, 0, "not a chunk", "")
        return ExecResult(True, 0, _chunk_response(reply, "c" * 32, 0, 100), "")

    sandbox = CannedSandbox(policy)
    transport = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)
    with override_max_exec_output_size(64), pytest.raises(RuntimeError) as raised:
        await transport("big", {}, False)

    assert not isinstance(raised.value, SandboxReplacedError)


async def test_sandbox_json_rpc_transport_leaves_plain_response() -> None:
    response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"})

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            assert json.loads(kwargs["input"])["method"] == "plain"
            return ExecResult(True, 0, response, "")

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    assert await transport("plain", {}, False) == response


async def test_sandbox_json_rpc_transport_leaves_json_rpc_errors() -> None:
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32099, "message": "tool failed"},
        }
    )

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            return ExecResult(True, 0, response, "")

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    assert await transport("plain", {}, False) == response


async def test_sandbox_json_rpc_transport_rejects_out_of_order_chunk() -> None:
    original_bytes = b'{"jsonrpc":"2.0","id":1,"result":"abcdefghijklmno"}'
    handle = "b" * 32
    continuation_count = 0

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            nonlocal continuation_count
            request = json.loads(kwargs["input"])
            if request["method"] != _CHUNK_METHOD:
                return ExecResult(
                    True, 0, _chunk_response(original_bytes, handle, 0, 10), ""
                )
            if request["params"].get("release") is True:
                return ExecResult(True, 0, _success_response(), "")
            continuation_count += 1
            offset = request["params"]["offset"] + 1
            return ExecResult(
                True, 0, _chunk_response(original_bytes, handle, offset, 10), ""
            )

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="out of order"):
        await transport("large", {}, False)
    assert continuation_count == 1


async def test_sandbox_json_rpc_transport_surfaces_continuation_error() -> None:
    original_bytes = b'{"jsonrpc":"2.0","id":1,"result":"abcdefghijklmno"}'
    handle = "c" * 32

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            request = json.loads(kwargs["input"])
            if request["method"] != _CHUNK_METHOD:
                return ExecResult(
                    True, 0, _chunk_response(original_bytes, handle, 0, 10), ""
                )
            if request["params"].get("release") is True:
                return ExecResult(True, 0, _success_response(), "")
            return ExecResult(
                True,
                0,
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "error": {"code": -32000, "message": "chunk missing"},
                    }
                ),
                "",
            )

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="chunk missing"):
        await transport("large", {}, False)


async def test_sandbox_json_rpc_transport_keeps_concurrent_handles_separate() -> None:
    responses = {
        "one": json.dumps({"jsonrpc": "2.0", "id": 1, "result": "one" * 200}),
        "two": json.dumps({"jsonrpc": "2.0", "id": 2, "result": "two" * 200}),
    }
    handles = {"one": "1" * 32, "two": "2" * 32}
    response_for_handle = {
        handles[name]: response.encode("utf-8") for name, response in responses.items()
    }

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            await anyio.sleep(0)
            request = json.loads(kwargs["input"])
            if request["method"] == _CHUNK_METHOD:
                params = request["params"]
                if params.get("release") is True:
                    return ExecResult(True, 0, _success_response(), "")
                handle = params["handle"]
                return ExecResult(
                    True,
                    0,
                    _chunk_response(
                        response_for_handle[handle], handle, params["offset"], 31
                    ),
                    "",
                )
            handle = handles[request["method"]]
            return ExecResult(
                True,
                0,
                _chunk_response(response_for_handle[handle], handle, 0, 31),
                "",
            )

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]
    results = await tg_collect(
        [
            partial(transport, "one", {}, False),
            partial(transport, "two", {}, False),
        ]
    )

    assert results == [responses["one"], responses["two"]]


async def test_legacy_sandbox_transport_does_not_enable_chunk_protocol() -> None:
    response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "legacy"})

    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            assert kwargs["env"] is None
            return ExecResult(True, 0, response, "")

    transport = SandboxJSONRPCTransport(FakeSandbox(), "inspect-tool-support")  # type: ignore[arg-type]

    assert await transport("legacy", {}, False) == response


async def test_sandbox_tools_cli_is_launched_after_the_directory_check() -> None:
    sandbox = CannedSandbox.returning(ExecResult(True, 0, _success_response(), ""))
    for cli in (SANDBOX_CLI, "inspect-tool-support"):
        await SandboxJSONRPCTransport(sandbox, cli)("m", {}, False)

    assert [cmd for cmd, _ in sandbox.exec_calls] == [
        tools_cli_command("exec", user=None),
        ["inspect-tool-support", "exec"],
    ]


async def test_a_reply_that_carries_the_refusal_marker_is_a_reply() -> None:
    """Only the check's own output, the marker alone, is a refusal."""
    reply = json.dumps({"jsonrpc": "2.0", "id": 1, "result": TOOLS_DIR_REFUSED})
    sandbox = CannedSandbox.returning(ExecResult(True, 0, reply, ""))
    assert await SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)("m", {}, False) == reply


async def test_sandbox_exec_error_uses_stdout_when_stderr_is_empty() -> None:
    class FakeSandbox:
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult[str]:
            return ExecResult(False, 1, "stdout diagnostic", "")

    transport = SandboxJSONRPCTransport(FakeSandbox(), SANDBOX_CLI)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="stdout diagnostic"):
        await transport("failing", {}, False)


def _chunk_response(
    response: bytes,
    handle: str,
    offset: int,
    chunk_size: int,
) -> str:
    chunk = response[offset : offset + chunk_size]
    next_offset = offset + len(chunk)
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            _CHUNK_FIELD: {
                "version": 1,
                "handle": handle,
                "offset": offset,
                "next_offset": next_offset,
                "total_size": len(response),
                "done": next_offset == len(response),
                "chunk": base64.b64encode(chunk).decode("ascii"),
            },
        }
    )


def _success_response() -> str:
    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": None})


# ============================================================================
# Bridge: many requests over one exec_stdio stream
# ============================================================================

_READY_LINE = (
    b'{"jsonrpc":"2.0","method":"__inspect_bridge_ready__","params":{"version":1}}\n'
)
_COMPLETED = {
    "pid": 42,
    "result": {
        "state": "completed",
        "exit_code": 0,
        "seq": 1,
        "stdout": "",
        "stderr": "",
    },
}
"""An exec_remote start's reply for a command that has exited."""


class _Pipe(ByteStream):
    """One end of an in-memory byte pipe."""

    def __init__(
        self,
        send: MemoryObjectSendStream[bytes],
        receive: MemoryObjectReceiveStream[bytes],
        peer_send: MemoryObjectSendStream[bytes],
    ) -> None:
        self._send = send
        self._receive = receive
        self._peer_send = peer_send  # closed with this end, to wake its reader
        self.close_fails = False
        self.send_fails = False
        self.receive_fails = False
        self.write_seconds = 0.0

    async def send(self, item: bytes) -> None:
        if self.send_fails:
            raise RuntimeError("the connection failed")
        # a byte stream may deliver one write in pieces
        half = len(item) // 2
        await self._send.send(item[:half])
        await anyio.sleep(self.write_seconds)
        await self._send.send(item[half:])

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self.receive_fails:
            raise RuntimeError("the connection failed")
        return await self._receive.receive()

    async def send_eof(self) -> None:
        await self._send.aclose()

    async def aclose(self) -> None:
        if self.close_fails:
            raise OSError("closing failed")
        await self._send.aclose()
        await self._peer_send.aclose()
        await self._receive.aclose()


class _FakeBridge:
    """The sandbox end of one bridge stream."""

    def __init__(self, stream: _Pipe, host: _Pipe) -> None:
        self.stream = stream
        self.host = host  # the end the host was given
        self.requests: list[dict[str, Any]] = []
        self._buffer = b""

    async def ready(self) -> None:
        await self.stream.send(_READY_LINE)

    async def next_request(self) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            self._buffer += await self.stream.receive()
        line, self._buffer = self._buffer.split(b"\n", 1)
        self.requests.append(json.loads(line))
        return self.requests[-1]

    async def reply(self, request: dict[str, Any], result: object = "bridged") -> None:
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        await self.stream.send(json.dumps(reply).encode() + b"\n")

    async def wait_closed(self) -> None:
        with contextlib.suppress(anyio.EndOfStream):
            while True:
                await self.stream.receive()


@sandboxenv(name="bridge_fake")
class _BridgeSandbox(SandboxEnvironment):
    """A sandbox whose `exec_stdio` runs `serve` on the other end of a pipe."""

    def __init__(
        self,
        tg: TaskGroup,
        serve: Callable[[_FakeBridge], Coroutine[Any, Any, None]] | None,
    ) -> None:
        super().__init__()
        self._tg = tg
        self._serve = serve
        self._root_access = RootAccess("unusable", "a test sandbox")
        self.stdin_buffer = 100
        self.before_open: Callable[[], Coroutine[Any, Any, None]] | None = None
        self.close_fails = False
        self.bridges: list[_FakeBridge] = []
        self.stdio_calls: list[list[str]] = []
        self.stdio_users: list[str | None] = []
        self.execs: list[str] = []
        self.cleanups: list[bool] = []

    @override
    async def exec_stdio(self, cmd: list[str], user: str | None = None) -> ByteStream:
        self.stdio_calls.append(cmd)
        self.stdio_users.append(user)
        if self._serve is None:
            return await super().exec_stdio(cmd, user)  # the default: not implemented
        if self.before_open is not None:
            await self.before_open()
        host, sandbox = _pipe(self.stdin_buffer)
        host.close_fails = self.close_fails
        self.bridges.append(_FakeBridge(sandbox, host))
        self._tg.start_soon(self._serve, self.bridges[-1])
        return host

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        assert isinstance(input, str)
        request = json.loads(input)
        self.execs.append(request["method"])
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": "per-call"}
        return ExecResult(True, 0, json.dumps(reply), "")

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        raise NotImplementedError

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        raise NotImplementedError

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        assert _sample_sandbox is not None
        return {"default": _sample_sandbox}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            environment.as_type(_BridgeSandbox).cleanups.append(interrupted)


_sample_sandbox: _BridgeSandbox | None = None
"""What a sample using the `bridge_fake` sandbox gets."""


def _pipe(host_to_sandbox_buffer: int) -> tuple[_Pipe, _Pipe]:
    host_send, sandbox_receive = anyio.create_memory_object_stream[bytes](
        host_to_sandbox_buffer
    )
    sandbox_send, host_receive = anyio.create_memory_object_stream[bytes](100)
    return (
        _Pipe(host_send, host_receive, sandbox_send),
        _Pipe(sandbox_send, sandbox_receive, host_send),
    )


@contextlib.asynccontextmanager
async def _bridge_sandbox(
    serve: Callable[[_FakeBridge], Coroutine[Any, Any, None]] | None,
) -> AsyncIterator[_BridgeSandbox]:
    """A `_BridgeSandbox` whose fake bridges are cancelled when the block ends."""
    async with anyio.create_task_group() as tg:
        yield _BridgeSandbox(tg, serve)
        tg.cancel_scope.cancel()


async def _serve_forever(bridge: _FakeBridge) -> None:
    await bridge.ready()
    with contextlib.suppress(anyio.EndOfStream, anyio.ClosedResourceError):
        while True:
            await bridge.reply(await bridge.next_request())


def _bridged_transport(sandbox: _BridgeSandbox) -> SandboxJSONRPCTransport:
    return SandboxJSONRPCTransport(sandbox, SANDBOX_CLI, bridge=True)


_BUDGET = 30
"""Seconds a test gives a request or a wait: a CI runner can stall for 5-13 s."""


async def _result(transport: SandboxJSONRPCTransport, method: str, **args: Any) -> Any:
    args.setdefault(
        "timeout", _BUDGET
    )  # fail rather than hang when a reply never comes
    return json.loads(await transport(method, {}, False, **args))["result"]


@pytest.fixture
def bridge_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_EXEC_REMOTE_BRIDGE", "1")


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_matches_out_of_order_replies_by_id() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        first = await bridge.next_request()
        second = await bridge.next_request()
        await bridge.reply(second, second["method"])
        await bridge.reply(first, first["method"])
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        results = await tg_collect(
            [
                partial(_result, transport, "exec_remote_start_wait"),
                partial(_result, transport, "exec_remote_poll_wait"),
            ]
        )

    assert results == ["exec_remote_start_wait", "exec_remote_poll_wait"]
    assert sandbox.execs == []


@pytest.mark.usefixtures("bridge_on")
async def test_request_over_an_open_bridge_is_sent_without_a_turn_of_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each turn of a busy runner's event loop is a burst of other work."""
    turns = 0
    turns_at_send: list[int] = []
    send = _Pipe.send

    async def counted_send(self: _Pipe, item: bytes) -> None:
        turns_at_send.append(turns)
        await send(self, item)

    async def other_work() -> None:
        nonlocal turns
        while True:
            turns += 1
            await anyio.lowlevel.checkpoint()

    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        await _result(transport, "exec_remote_poll_wait")  # opens the bridge
        monkeypatch.setattr(_Pipe, "send", counted_send)
        async with anyio.create_task_group() as tg:
            tg.start_soon(other_work)
            await anyio.lowlevel.checkpoint()  # it runs once
            before = turns
            await _result(transport, "exec_remote_poll_wait")
            tg.cancel_scope.cancel()

    assert turns_at_send[0] == before


@pytest.mark.usefixtures("bridge_on")
async def test_exec_remote_over_an_open_bridge_is_sent_without_a_turn_of_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tools are checked once per sandbox, so a later call goes straight out."""
    turns = 0
    turns_at_send: list[int] = []
    send = _Pipe.send

    async def counted_send(self: _Pipe, item: bytes) -> None:
        turns_at_send.append(turns)
        await send(self, item)

    async def other_work() -> None:
        nonlocal turns
        while True:
            turns += 1
            await anyio.lowlevel.checkpoint()

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        while True:
            await bridge.reply(await bridge.next_request(), _COMPLETED)

    async with _bridge_sandbox(serve) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        proxy._tools_injected = True  # checked earlier in the sample
        options = ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET)
        await proxy.exec_remote(["ls"], options, stream=False)  # opens the bridge
        monkeypatch.setattr(_Pipe, "send", counted_send)
        async with anyio.create_task_group() as tg:
            tg.start_soon(other_work)
            await anyio.lowlevel.checkpoint()  # it runs once
            before = turns
            await proxy.exec_remote(["ls"], options, stream=False)
            tg.cancel_scope.cancel()

    assert turns_at_send[0] == before


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_resends_a_request_lost_with_a_dropped_bridge() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) == 1:
            await bridge.stream.aclose()
            return
        await bridge.reply(request)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert result == "bridged"
    first, second = sandbox.bridges
    assert first.requests == second.requests
    assert sandbox.execs == []


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_raises_when_the_resent_request_is_lost_too() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.next_request()
        await bridge.stream.aclose()

    async with _bridge_sandbox(serve) as sandbox:
        with pytest.raises(BridgeDropped, match="the bridge exited"):
            await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert len(sandbox.bridges) == 2
    assert sandbox.execs == []


@pytest.mark.parametrize(
    "whole_line", [False, True], ids=["no newline within the limit", "a valid reply"]
)
@pytest.mark.usefixtures("bridge_on")
async def test_bridge_reply_longer_than_the_frame_limit_drops_the_bridge(
    whole_line: bool,
) -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) > 1:
            await bridge.reply(request)
        elif whole_line:  # arrives in one read
            await bridge.reply(request, "x" * 2000)
        else:
            await bridge.stream.send(b"x" * 1024)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        with override_max_exec_output_size(1024):
            result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert result == "bridged"
    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_sandbox_without_exec_stdio_sends_each_request_in_its_own_exec() -> None:
    async with _bridge_sandbox(None) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_start_wait") == "per-call"
        assert await _result(transport, "exec_remote_poll_wait") == "per-call"

    assert len(sandbox.stdio_calls) == 1
    assert sandbox.execs == ["exec_remote_start_wait", "exec_remote_poll_wait"]


@pytest.mark.parametrize(
    "first_line",
    [None, b"usage: inspect-sandbox-tools ...\n"],
    ids=["exits", "prints something else"],
)
@pytest.mark.usefixtures("bridge_on")
async def test_bridge_that_does_not_start_is_not_started_again(
    first_line: bytes | None,
) -> None:
    async def serve(bridge: _FakeBridge) -> None:
        # e.g. an older sandbox tools build, which has no `bridge` command
        if first_line is not None:
            await bridge.stream.send(first_line)
            await bridge.wait_closed()
        await bridge.stream.aclose()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_start_wait") == "per-call"
        assert await _result(transport, "exec_remote_poll_wait") == "per-call"

    assert len(sandbox.bridges) == 1
    assert sandbox.execs == ["exec_remote_start_wait", "exec_remote_poll_wait"]


@pytest.mark.usefixtures("bridge_on")
async def test_request_that_times_out_closes_the_bridge() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) > 1:
            await bridge.reply(request)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        with pytest.raises(SandboxTimeoutError, match="exec_remote_poll_wait"):
            await _result(transport, "exec_remote_poll_wait", timeout=0.05)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_request_that_cannot_be_written_in_time_times_out() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        if len(sandbox.bridges) == 1:
            await bridge.ready()
            await anyio.sleep_forever()  # never reads its stdin
        await _serve_forever(bridge)

    async with _bridge_sandbox(serve) as sandbox:
        sandbox.stdin_buffer = 0
        transport = _bridged_transport(sandbox)
        with pytest.raises(SandboxTimeoutError):
            await _result(transport, "exec_remote_poll_wait", timeout=0.05)
        sandbox.stdin_buffer = 100
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_that_does_not_open_in_time_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(json_rpc_bridge, "OPEN_TIMEOUT", 0.05)
    async with _bridge_sandbox(_serve_forever) as sandbox:
        sandbox.before_open = anyio.sleep_forever
        transport = _bridged_transport(sandbox)
        with anyio.fail_after(_BUDGET):  # the request's timeout does not cover opening
            assert await _result(transport, "exec_remote_poll_wait") == "per-call"
            assert await _result(transport, "exec_remote_poll_wait") == "per-call"

    assert len(sandbox.stdio_calls) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_provider_error_starting_a_bridge_is_raised_once() -> None:
    async def replaced() -> None:
        await anyio.sleep(0.05)
        raise RuntimeError("the pod was replaced")

    results: list[Any] = []

    async def request(transport: SandboxJSONRPCTransport) -> None:
        try:
            results.append(await _result(transport, "exec_remote_poll_wait"))
        except RuntimeError as ex:
            results.append(str(ex))

    async with _bridge_sandbox(_serve_forever) as sandbox:
        sandbox.before_open = replaced
        transport = _bridged_transport(sandbox)
        async with anyio.create_task_group() as requests:
            for _ in range(3):
                requests.start_soon(request, transport)

    assert sorted(results) == ["per-call", "per-call", "the pod was replaced"]
    assert len(sandbox.stdio_calls) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_error_reply_raises_like_a_failed_exec() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        reply = {
            "jsonrpc": "2.0",
            "id": request["id"],
            "__inspect_bridge_error__": "ConnectionError: server closed the connection",
        }
        await bridge.stream.send(json.dumps(reply).encode() + b"\n")
        await bridge.wait_closed()

    init_transcript(Transcript())
    async with _bridge_sandbox(serve) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        transport = SandboxJSONRPCTransport(proxy, SANDBOX_CLI, bridge=True)
        with pytest.raises(RuntimeError, match="Sandbox.exec failure executing"):
            await _result(transport, "exec_remote_start_wait")

    [event] = [e for e in transcript().events if isinstance(e, SandboxEvent)]
    assert (event.cmd, event.result) == (f"{SANDBOX_CLI} bridge", 1)
    assert "server closed the connection" in (event.output or "")


@pytest.mark.usefixtures("bridge_on")
async def test_only_the_bridge_methods_use_the_bridge() -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        for method in ["exec_remote_start", "exec_remote_write_stdin", "bash_session"]:
            assert await _result(transport, method) == "per-call"
        assert await _result(transport, "exec_remote_kill_wait") == "bridged"
        unbridged = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)
        assert await _result(unbridged, "exec_remote_kill_wait") == "per-call"

    assert sandbox.execs == [
        "exec_remote_start",
        "exec_remote_write_stdin",
        "bash_session",
        "exec_remote_kill_wait",
    ]


@pytest.mark.parametrize(
    "value,bridged",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        (" True ", True),
    ],
)
async def test_bridge_is_on_only_when_enabled(
    value: str | None, bridged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv("INSPECT_EXEC_REMOTE_BRIDGE", raising=False)
    else:
        monkeypatch.setenv("INSPECT_EXEC_REMOTE_BRIDGE", value)
    async with _bridge_sandbox(_serve_forever) as sandbox:
        result = await _result(_bridged_transport(sandbox), "exec_remote_start_wait")

    assert result == ("bridged" if bridged else "per-call")


@pytest.mark.usefixtures("bridge_on")
async def test_request_too_long_for_a_frame_is_sent_in_its_own_exec() -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        with override_max_exec_output_size(1024):
            reply = await transport(
                "exec_remote_start_wait", {"input": "x" * 1024}, False
            )
        assert json.loads(reply)["result"] == "per-call"

    assert sandbox.stdio_calls == []


@pytest.mark.usefixtures("bridge_on")
async def test_cancelled_reader_loses_no_reply() -> None:
    first_sent = anyio.Event()
    second_sent = anyio.Event()
    first_cancelled = anyio.Event()
    results: list[Any] = []

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.next_request()
        first_sent.set()
        second = await bridge.next_request()
        reply = json.dumps({"jsonrpc": "2.0", "id": second["id"], "result": "second"})
        # half of the reply reaches the first request, which is reading
        await bridge.stream.send(reply[:10].encode())
        second_sent.set()
        await first_cancelled.wait()
        await bridge.stream.send(reply[10:].encode() + b"\n")
        await bridge.wait_closed()

    async def first(
        transport: SandboxJSONRPCTransport, scope: anyio.CancelScope
    ) -> None:
        with scope:
            await _result(transport, "exec_remote_poll_wait")

    async def second(transport: SandboxJSONRPCTransport) -> None:
        results.append(await _result(transport, "exec_remote_poll_wait"))

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        scope = anyio.CancelScope()
        async with anyio.create_task_group() as requests:
            requests.start_soon(first, transport, scope)
            await first_sent.wait()
            requests.start_soon(second, transport)
            await second_sent.wait()
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
            first_cancelled.set()

    assert scope.cancelled_caught
    assert results == ["second"]
    assert len(sandbox.bridges) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_close_bridges_ends_the_bridge_for_good() -> None:
    closed = anyio.Event()

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.reply(await bridge.next_request())
        await bridge.wait_closed()
        closed.set()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_start_wait") == "bridged"
        await close_bridges(sandbox)
        with anyio.fail_after(_BUDGET):
            await closed.wait()
        assert await _result(transport, "exec_remote_start_wait") == "per-call"

    assert len(sandbox.bridges) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_bridged_request_records_the_exec_event() -> None:
    init_transcript(Transcript())
    async with _bridge_sandbox(_serve_forever) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        transport = SandboxJSONRPCTransport(proxy, SANDBOX_CLI, bridge=True)
        await transport(
            "exec_remote_start_wait",
            {"command": "ls"},
            False,
            user="root",
            timeout=_BUDGET,
        )

    [event] = [e for e in transcript().events if isinstance(e, SandboxEvent)]
    assert (event.action, event.cmd, event.result) == (
        "exec",
        f"{SANDBOX_CLI} bridge",
        0,
    )
    assert json.loads(event.input or "")["params"] == {"command": "ls"}
    assert event.options == {"user": "root", "timeout": _BUDGET}


@pytest.mark.usefixtures("bridge_on")
async def test_exec_remote_runs_over_the_bridge() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        await bridge.reply(
            request,
            {
                "pid": 42,
                "result": {
                    "state": "completed",
                    "exit_code": 3,
                    "seq": 1,
                    "stdout": "out",
                    "stderr": "err",
                },
            },
        )
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        result = await exec_remote_awaitable(
            proxy,
            ["ls"],
            5,
            ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET),
        )

    assert (result.returncode, result.stdout, result.stderr) == (3, "out", "err")
    [bridge] = sandbox.bridges
    assert [r["method"] for r in bridge.requests] == ["exec_remote_start_wait"]
    assert sandbox.execs == []


@pytest.mark.parametrize("ending", ["\n", "\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("calls_before", [0, 1], ids=["first open", "reopen"])
@pytest.mark.usefixtures("bridge_on")
async def test_exec_remote_fails_as_a_replaced_sandbox_when_its_bridge_is_refused(
    calls_before: int, ending: str
) -> None:
    """The tools directory went away since the install check, so the launch is refused.

    On a reopen, the bridge before it ended with its container. The refusal line is
    compared stripped, as the exec path compares stdout.
    """
    refused_closed = anyio.Event()

    async def serve(bridge: _FakeBridge) -> None:
        if len(sandbox.bridges) <= calls_before:
            await bridge.ready()
            await bridge.reply(await bridge.next_request(), _COMPLETED)
            await bridge.stream.aclose()  # its container restarted
        else:
            await bridge.stream.send(f"{TOOLS_DIR_REFUSED}{ending}".encode())
            await bridge.wait_closed()  # left to the host to close
            refused_closed.set()

    async with _bridge_sandbox(serve) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        options = ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET)
        for _ in range(calls_before):
            assert (await exec_remote_awaitable(proxy, ["ls"], 5, options)).success
        with pytest.raises(SandboxReplacedError, match="the sandbox tools bridge"):
            await exec_remote_awaitable(proxy, ["ls"], 5, options)
        with anyio.fail_after(_BUDGET):
            await refused_closed.wait()

    # not taken for an older build, nor resent or killed in an exec
    assert (len(sandbox.stdio_calls), sandbox.execs) == (calls_before + 1, [])


@pytest.mark.usefixtures("bridge_on")
async def test_each_frame_limit_gets_its_own_bridge() -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        await _result(transport, "exec_remote_poll_wait")
        with override_max_exec_output_size(4096):
            await _result(transport, "exec_remote_poll_wait")
        await _result(transport, "exec_remote_poll_wait")

    assert [cmd[-1] for cmd in sandbox.stdio_calls] == ["10485760", "4096"]
    assert [len(bridge.requests) for bridge in sandbox.bridges] == [2, 1]


@pytest.mark.parametrize(
    "cleanup,close_fails",
    [(True, False), (False, False), (True, True)],
    ids=["cleaned up", "kept", "close fails"],
)
@pytest.mark.usefixtures("bridge_on")
async def test_sample_exit_closes_its_bridges(
    cleanup: bool, close_fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = anyio.Event()

    async def serve(bridge: _FakeBridge) -> None:
        await _serve_forever(bridge)
        closed.set()

    async with _bridge_sandbox(serve) as provider:
        provider.close_fails = close_fails
        monkeypatch.setattr(sys.modules[__name__], "_sample_sandbox", provider)
        async with sandboxenv_context(
            "task",
            SandboxEnvironmentSpec("bridge_fake"),
            None,
            cleanup,
            Sample(input="x"),
        ):
            transport = SandboxJSONRPCTransport(
                current_sandbox(), SANDBOX_CLI, bridge=True
            )
            assert await _result(transport, "exec_remote_start_wait") == "bridged"
        if not close_fails:
            with anyio.fail_after(_BUDGET):
                await closed.wait()

    assert provider.cleanups == ([False] if cleanup else [])


@pytest.mark.parametrize(
    "reply",
    [b'{"jsonrpc": "2.0", "id": 9999, "result": 1}', b"[" * 100_000],
    ids=["unknown ID", "too deeply nested to parse"],
)
@pytest.mark.usefixtures("bridge_on")
async def test_bridge_reply_no_request_waits_for_drops_the_bridge(reply: bytes) -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) == 1:
            await bridge.stream.send(reply + b"\n")
        else:
            await bridge.reply(request)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        assert (
            await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")
            == "bridged"
        )

    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_late_reply_to_a_request_that_timed_out_is_dropped() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        late = await bridge.next_request()
        request = await bridge.next_request()
        await bridge.reply(late)
        await bridge.reply(request)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        with anyio.move_on_after(0.05):
            await _result(transport, "exec_remote_poll_wait")
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert len(sandbox.bridges) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_each_tools_user_gets_its_own_bridge() -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        await _result(transport, "exec_remote_poll_wait", user="root")
        await _result(transport, "exec_remote_poll_wait", user=None)
        await _result(transport, "exec_remote_poll_wait", user="root")

    assert sandbox.stdio_users == ["root", None]
    assert sandbox.stdio_calls == [
        tools_cli_command("bridge", "--max-frame-bytes", "10485760", user=user)
        for user in ("root", None)
    ]
    assert [len(bridge.requests) for bridge in sandbox.bridges] == [2, 1]


@pytest.mark.parametrize("serve_ready", [False, True], ids=["exits", "never ready"])
@pytest.mark.usefixtures("bridge_on")
async def test_bridge_that_does_not_start_is_closed(
    serve_ready: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bridge that fails to report ready, or whose start is cancelled, is closed."""
    monkeypatch.setattr(json_rpc_bridge, "OPEN_TIMEOUT", 0.05)
    closed = anyio.Event()

    async def serve(bridge: _FakeBridge) -> None:
        if not serve_ready:  # cancelled below while it waits
            await anyio.sleep(0.01)
        await bridge.wait_closed()
        closed.set()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        with anyio.fail_after(_BUDGET):
            if serve_ready:
                assert await _result(transport, "exec_remote_poll_wait") == "per-call"
            else:
                with anyio.move_on_after(0.01):
                    await _result(transport, "exec_remote_poll_wait")
            await closed.wait()


@pytest.mark.usefixtures("bridge_on")
async def test_nothing_is_sent_before_the_bridge_is_ready() -> None:
    """A provider may start the bridge through a shell reading the same stdin.

    dash and busybox sh read ahead of the line they run, so a request sent before
    the bridge started could be lost to the shell.
    """
    early: list[bytes] = []

    async def serve(bridge: _FakeBridge) -> None:
        # Once every task waits, the host either waits for the ready line or has
        # sent its request already.
        await anyio.wait_all_tasks_blocked()
        with contextlib.suppress(anyio.WouldBlock):
            early.append(bridge.stream._receive.receive_nowait())
        await bridge.ready()
        await bridge.reply(await bridge.next_request())

    async with _bridge_sandbox(serve) as sandbox:
        result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert (early, result) == ([], "bridged")


@pytest.mark.usefixtures("bridge_on")
async def test_chunked_reply_is_fetched_over_the_bridge() -> None:
    original = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "x" * 2000}).encode()
    handle = "d" * 32

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        while True:
            request = await bridge.next_request()
            params = request.get("params", {})
            if params.get("release") is True:
                await bridge.reply(request, None)
                continue
            chunk = json.loads(
                _chunk_response(original, handle, params.get("offset", 0), 500)
            )
            chunk["id"] = request["id"]
            await bridge.stream.send(json.dumps(chunk).encode() + b"\n")

    async with _bridge_sandbox(serve) as sandbox:
        with override_max_exec_output_size(1024):
            reply = await _bridged_transport(sandbox)(
                "exec_remote_poll_wait", {}, False, timeout=_BUDGET
            )

    assert reply == original.decode()
    methods = [request["method"] for request in sandbox.bridges[0].requests]
    assert methods[0] == "exec_remote_poll_wait"
    assert set(methods[1:]) == {_CHUNK_METHOD}
    assert sandbox.execs == []


@pytest.mark.usefixtures("bridge_on")
async def test_held_request_over_an_open_bridge_ignores_the_held_request_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY", "0")
    completed = {
        "pid": 42,
        "result": {
            "state": "completed",
            "exit_code": 0,
            "seq": 1,
            "stdout": "",
            "stderr": "",
        },
    }

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        while True:
            await bridge.reply(await bridge.next_request(), completed)

    async with _bridge_sandbox(serve) as sandbox:
        proxy = SandboxEnvironmentProxy(sandbox)
        options = ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET)
        for _ in range(2):
            await exec_remote_awaitable(proxy, ["ls"], 5, options)

    first, second = (request["params"] for request in sandbox.bridges[0].requests)
    assert first["wait_seconds"] == 0.0  # no bridge was open yet
    assert second["wait_seconds"] > 0


@pytest.mark.parametrize("failing", ["send", "receive"])
@pytest.mark.usefixtures("bridge_on")
async def test_bridge_whose_stream_fails_is_replaced(failing: str) -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"
        setattr(sandbox.bridges[0].host, f"{failing}_fails", True)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert len(sandbox.bridges) == 2


@pytest.mark.parametrize("extra", [0, 1], ids=["exactly the limit", "one byte over"])
@pytest.mark.usefixtures("bridge_on")
async def test_reply_frame_limit_includes_the_newline(extra: int) -> None:
    sent: list[str] = []

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) > 1:
            await bridge.reply(request)
        else:
            reply = {"jsonrpc": "2.0", "id": request["id"], "result": ""}
            reply["result"] = "x" * (1024 - len(json.dumps(reply)) - 1 + extra)
            sent.append(reply["result"])
            await bridge.stream.send(json.dumps(reply).encode() + b"\n")
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        with override_max_exec_output_size(1024):
            result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert len(sandbox.bridges) == 1 + extra
    assert result == ("bridged" if extra else sent[0])


@pytest.mark.usefixtures("bridge_on")
async def test_closing_a_bridge_resends_the_requests_waiting_on_it() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        if len(sandbox.bridges) > 1:
            await _serve_forever(bridge)
            return
        await bridge.ready()
        await bridge.next_request()
        await bridge.next_request()
        await bridge.wait_closed()

    results: list[Any] = []

    async def waiting(transport: SandboxJSONRPCTransport) -> None:
        results.append(
            await _result(transport, "exec_remote_poll_wait", timeout=2 * _BUDGET)
        )

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        with anyio.fail_after(_BUDGET):  # well inside the waiting request's own timeout
            async with anyio.create_task_group() as requests:
                requests.start_soon(waiting, transport)
                await anyio.wait_all_tasks_blocked()
                with pytest.raises(SandboxTimeoutError):
                    await _result(transport, "exec_remote_poll_wait", timeout=0.05)

    assert results == ["bridged"]


@pytest.mark.usefixtures("bridge_on")
async def test_closing_a_bridge_fails_its_waiting_requests_even_if_its_stream_stays_open() -> (
    None
):
    async def serve(bridge: _FakeBridge) -> None:
        if len(sandbox.bridges) > 1:
            await _serve_forever(bridge)
            return
        await bridge.ready()
        await bridge.next_request()
        await bridge.next_request()
        await anyio.sleep_forever()  # answers nothing; closing its stream fails

    results: list[Any] = []

    async def waiting(transport: SandboxJSONRPCTransport) -> None:
        results.append(
            await _result(transport, "exec_remote_poll_wait", timeout=2 * _BUDGET)
        )

    async with _bridge_sandbox(serve) as sandbox:
        sandbox.close_fails = True
        transport = _bridged_transport(sandbox)
        with anyio.fail_after(
            _BUDGET
        ):  # well inside the waiting requests' own timeouts
            async with anyio.create_task_group() as requests:
                requests.start_soon(waiting, transport)
                requests.start_soon(waiting, transport)
                await anyio.wait_all_tasks_blocked()
                sandbox.close_fails = False  # only the first bridge's close fails
                with pytest.raises(SandboxTimeoutError):
                    await _result(transport, "exec_remote_poll_wait", timeout=0.05)

    assert results == ["bridged", "bridged"]


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_that_opens_after_the_sample_ends_is_closed() -> None:
    opening, release = anyio.Event(), anyio.Event()
    closed = anyio.Event()

    async def slow_open() -> None:
        opening.set()
        await release.wait()

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.wait_closed()
        closed.set()

    results: list[Any] = []

    async def request(transport: SandboxJSONRPCTransport) -> None:
        results.append(await _result(transport, "exec_remote_poll_wait"))

    async with _bridge_sandbox(serve) as sandbox:
        sandbox.before_open = slow_open
        transport = _bridged_transport(sandbox)
        with anyio.fail_after(_BUDGET):
            async with anyio.create_task_group() as requests:
                requests.start_soon(request, transport)
                await opening.wait()
                await close_bridges(sandbox)
                release.set()
            await closed.wait()

    assert results == ["per-call"]


@pytest.mark.usefixtures("bridge_on")
async def test_reply_for_an_id_never_sent_drops_the_bridge_after_an_abandoned_request() -> (
    None
):
    async def serve(bridge: _FakeBridge) -> None:
        if len(sandbox.bridges) > 1:
            await _serve_forever(bridge)
            return
        await bridge.ready()
        abandoned = await bridge.next_request()
        request = await bridge.next_request()
        stranger = {"jsonrpc": "2.0", "id": abandoned["id"] + 10_000, "result": 1}
        await bridge.stream.send(json.dumps(stranger).encode() + b"\n")
        with contextlib.suppress(anyio.ClosedResourceError):  # the host dropped it
            await bridge.reply(request)
        await bridge.wait_closed()

    async with _bridge_sandbox(serve) as sandbox:
        transport = _bridged_transport(sandbox)
        with anyio.move_on_after(0.05):
            await _result(transport, "exec_remote_poll_wait")
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_held_request_too_long_for_a_frame_takes_a_held_request_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY", "0")
    completed = {
        "pid": 42,
        "result": {
            "state": "completed",
            "exit_code": 0,
            "seq": 1,
            "stdout": "",
            "stderr": "",
        },
    }
    held: list[float] = []

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        while True:
            await bridge.reply(await bridge.next_request(), completed)

    async def exec(cmd: list[str], **kwargs: Any) -> ExecResult[str]:
        request = json.loads(kwargs["input"])
        held.append(request["params"]["wait_seconds"])
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": completed}
        return ExecResult(True, 0, json.dumps(reply), "")

    async with _bridge_sandbox(serve) as sandbox:
        monkeypatch.setattr(sandbox, "exec", exec)
        proxy = SandboxEnvironmentProxy(sandbox)
        with override_max_exec_output_size(4096):
            options = ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET)
            await exec_remote_awaitable(proxy, ["ls"], 5, options)  # opens the bridge
            options = ExecRemoteAwaitableOptions(
                input="x" * 8192, server_wait=True, poll_timeout=_BUDGET
            )
            await exec_remote_awaitable(proxy, ["cat"], 5, options)

    assert held == [0.0]  # sent in its own exec, with no slot free, so not held


@pytest.mark.usefixtures("bridge_on")
async def test_held_request_whose_bridge_cannot_restart_takes_a_held_request_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY", "0")
    completed = {
        "pid": 42,
        "result": {
            "state": "completed",
            "exit_code": 0,
            "seq": 1,
            "stdout": "",
            "stderr": "",
        },
    }
    held: list[float] = []

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.reply(await bridge.next_request(), completed)
        await bridge.next_request()
        await bridge.stream.aclose()  # exits with the held request unanswered

    async def exec(cmd: list[str], **kwargs: Any) -> ExecResult[str]:
        request = json.loads(kwargs["input"])
        held.append(request["params"]["wait_seconds"])
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": completed}
        return ExecResult(True, 0, json.dumps(reply), "")

    async def not_ready() -> None:
        raise TimeoutError

    async with _bridge_sandbox(serve) as sandbox:
        monkeypatch.setattr(sandbox, "exec", exec)
        proxy = SandboxEnvironmentProxy(sandbox)
        options = ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=_BUDGET)
        await exec_remote_awaitable(proxy, ["ls"], 5, options)  # opens the bridge
        sandbox.before_open = not_ready
        await exec_remote_awaitable(proxy, ["ls"], 5, options)

    assert sandbox.bridges[0].requests[1]["params"]["wait_seconds"] > 0
    assert held == [0.0]  # sent in its own exec, with no slot free, so not held


@pytest.mark.usefixtures("bridge_on")
async def test_request_cancelled_before_it_is_sent_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a request that went out can still get a reply to drop."""
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"
        host = sandbox.bridges[0].host
        may_write = anyio.Event()
        send = host.send

        async def held_send(item: bytes) -> None:
            await may_write.wait()
            await send(item)

        queued = anyio.CancelScope()

        async def request_queued_behind() -> None:
            with queued:
                await _result(transport, "exec_remote_poll_wait")

        monkeypatch.setattr(host, "send", held_send)
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_result, transport, "exec_remote_poll_wait")
                await anyio.wait_all_tasks_blocked()  # it holds the send lock
                tg.start_soon(request_queued_behind)
                await anyio.wait_all_tasks_blocked()  # it waits for the lock
                queued.cancel()
                await anyio.wait_all_tasks_blocked()
                may_write.set()
        bridges = sandbox._json_rpc_bridges

    assert bridges is not None
    assert [bridge._abandoned for bridge in bridges.by_key.values() if bridge] == [
        set()
    ]


@pytest.mark.usefixtures("bridge_on")
async def test_caller_cancelled_while_its_request_is_written_keeps_the_bridge() -> None:
    async with _bridge_sandbox(_serve_forever) as sandbox:
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"
        sandbox.bridges[0].host.write_seconds = 0.2
        with anyio.move_on_after(0.05) as cancelled:
            await _result(transport, "exec_remote_poll_wait")
        sandbox.bridges[0].host.write_seconds = 0.0
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"

    assert cancelled.cancelled_caught
    assert len(sandbox.bridges) == 1
