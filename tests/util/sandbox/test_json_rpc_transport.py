import base64
import contextlib
import json
from collections.abc import Callable, Coroutine
from functools import partial
from typing import Any

import anyio
import pytest
from anyio.abc import ByteStream, TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from inspect_ai._util._async import tg_collect
from inspect_ai.event import SandboxEvent
from inspect_ai.log._transcript import Transcript, init_transcript, transcript
from inspect_ai.util._sandbox import context as sandbox_context
from inspect_ai.util._sandbox._cli import SANDBOX_CLI
from inspect_ai.util._sandbox._json_rpc_bridge import BridgeDropped, close_bridges
from inspect_ai.util._sandbox._json_rpc_transport import SandboxJSONRPCTransport
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy, SandboxTimeoutError
from inspect_ai.util._sandbox.exec_remote import (
    ExecRemoteAwaitableOptions,
    exec_remote_awaitable,
)
from inspect_ai.util._sandbox.limits import override_max_exec_output_size
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


class _Pipe(ByteStream):
    """One end of an in-memory byte pipe."""

    def __init__(
        self,
        send: MemoryObjectSendStream[bytes],
        receive: MemoryObjectReceiveStream[bytes],
    ) -> None:
        self._send = send
        self._receive = receive

    async def send(self, item: bytes) -> None:
        await self._send.send(item)

    async def receive(self, max_bytes: int = 65536) -> bytes:
        return await self._receive.receive()

    async def send_eof(self) -> None:
        await self._send.aclose()

    async def aclose(self) -> None:
        await self._send.aclose()
        await self._receive.aclose()


class _FakeBridge:
    """The sandbox end of one bridge stream."""

    def __init__(self, stream: _Pipe) -> None:
        self.stream = stream
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


class _BridgeSandbox:
    """A sandbox whose `exec_stdio` runs `serve` on the other end of a pipe."""

    _json_rpc_bridges = None

    def __init__(
        self,
        tg: TaskGroup,
        serve: Callable[[_FakeBridge], Coroutine[Any, Any, None]] | None,
    ) -> None:
        self._tg = tg
        self._serve = serve
        self.bridges: list[_FakeBridge] = []
        self.stdio_calls: list[list[str]] = []
        self.execs: list[str] = []

    async def exec_stdio(self, cmd: list[str], user: str | None = None) -> _Pipe:
        self.stdio_calls.append(cmd)
        if self._serve is None:
            raise NotImplementedError("exec_stdio not implemented")
        host, sandbox = _pipe()
        self.bridges.append(_FakeBridge(sandbox))
        self._tg.start_soon(self._serve, self.bridges[-1])
        return host

    async def exec(self, cmd: list[str], **kwargs: Any) -> ExecResult[str]:
        request = json.loads(kwargs["input"])
        self.execs.append(request["method"])
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": "per-call"}
        return ExecResult(True, 0, json.dumps(reply), "")


def _pipe() -> tuple[_Pipe, _Pipe]:
    host_send, sandbox_receive = anyio.create_memory_object_stream[bytes](100)
    sandbox_send, host_receive = anyio.create_memory_object_stream[bytes](100)
    return _Pipe(host_send, host_receive), _Pipe(sandbox_send, sandbox_receive)


async def _serve_forever(bridge: _FakeBridge) -> None:
    await bridge.ready()
    with contextlib.suppress(anyio.EndOfStream):
        while True:
            await bridge.reply(await bridge.next_request())


def _bridged_transport(sandbox: _BridgeSandbox) -> SandboxJSONRPCTransport:
    return SandboxJSONRPCTransport(sandbox, SANDBOX_CLI, bridge=True)  # type: ignore[arg-type]


async def _result(transport: SandboxJSONRPCTransport, method: str, **args: Any) -> Any:
    args.setdefault("timeout", 10)  # fail rather than hang when a reply never comes
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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        transport = _bridged_transport(sandbox)
        results = await tg_collect(
            [
                partial(_result, transport, "exec_remote_start_wait"),
                partial(_result, transport, "exec_remote_poll_wait"),
            ]
        )
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert results == ["exec_remote_start_wait", "exec_remote_poll_wait"]
    assert sandbox.stdio_calls == [
        [SANDBOX_CLI, "bridge", "--max-frame-bytes", "10485760"]
    ]
    assert sandbox.execs == []


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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")
        await close_bridges(sandbox)  # type: ignore[arg-type]

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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        with pytest.raises(BridgeDropped, match="the bridge exited"):
            await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")

    assert len(sandbox.bridges) == 2
    assert sandbox.execs == []


@pytest.mark.usefixtures("bridge_on")
async def test_bridge_reply_longer_than_the_frame_limit_drops_the_bridge() -> None:
    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        request = await bridge.next_request()
        if len(sandbox.bridges) == 1:
            await bridge.stream.send(b"x" * 1024)
        else:
            await bridge.reply(request)
        await bridge.wait_closed()

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        with override_max_exec_output_size(1024):
            result = await _result(_bridged_transport(sandbox), "exec_remote_poll_wait")
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert result == "bridged"
    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_sandbox_without_exec_stdio_sends_each_request_in_its_own_exec() -> None:
    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, None)
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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        transport = _bridged_transport(sandbox)
        with pytest.raises(SandboxTimeoutError, match="exec_remote_poll_wait"):
            await _result(transport, "exec_remote_poll_wait", timeout=0.05)
        assert await _result(transport, "exec_remote_poll_wait") == "bridged"
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert len(sandbox.bridges) == 2


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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        with pytest.raises(RuntimeError, match="Sandbox.exec failure executing"):
            await _result(_bridged_transport(sandbox), "exec_remote_start_wait")
        await close_bridges(sandbox)  # type: ignore[arg-type]


@pytest.mark.usefixtures("bridge_on")
async def test_only_requests_naming_a_request_id_use_the_bridge() -> None:
    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, _serve_forever)
        transport = _bridged_transport(sandbox)
        for method in ["exec_remote_start", "exec_remote_write_stdin", "bash_session"]:
            assert await _result(transport, method) == "per-call"
        assert await _result(transport, "exec_remote_kill_wait") == "bridged"
        unbridged = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)  # type: ignore[arg-type]
        assert await _result(unbridged, "exec_remote_kill_wait") == "per-call"
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert sandbox.execs == [
        "exec_remote_start",
        "exec_remote_write_stdin",
        "bash_session",
        "exec_remote_kill_wait",
    ]


async def test_bridge_is_off_unless_enabled() -> None:
    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, _serve_forever)
        assert (
            await _result(_bridged_transport(sandbox), "exec_remote_start_wait")
            == "per-call"
        )

    assert sandbox.stdio_calls == []


@pytest.mark.usefixtures("bridge_on")
async def test_request_too_long_for_a_frame_is_sent_in_its_own_exec() -> None:
    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, _serve_forever)
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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        transport = _bridged_transport(sandbox)
        scope = anyio.CancelScope()
        tg.start_soon(first, transport, scope)
        await first_sent.wait()
        async with anyio.create_task_group() as requests:
            requests.start_soon(second, transport)
            await second_sent.wait()
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
            first_cancelled.set()
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert scope.cancelled_caught
    assert results == ["second"]
    assert len(sandbox.bridges) == 1


@pytest.mark.usefixtures("bridge_on")
async def test_close_bridges_ends_the_bridge() -> None:
    closed = anyio.Event()

    async def serve(bridge: _FakeBridge) -> None:
        await bridge.ready()
        await bridge.reply(await bridge.next_request())
        await bridge.wait_closed()
        closed.set()

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        transport = _bridged_transport(sandbox)
        assert await _result(transport, "exec_remote_start_wait") == "bridged"
        await close_bridges(sandbox)  # type: ignore[arg-type]
        with anyio.fail_after(10):
            await closed.wait()
        assert await _result(transport, "exec_remote_start_wait") == "bridged"
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert len(sandbox.bridges) == 2


@pytest.mark.usefixtures("bridge_on")
async def test_bridged_request_records_the_exec_event() -> None:
    init_transcript(Transcript())
    async with anyio.create_task_group() as tg:
        proxy = SandboxEnvironmentProxy(_BridgeSandbox(tg, _serve_forever))  # type: ignore[arg-type]
        transport = SandboxJSONRPCTransport(proxy, SANDBOX_CLI, bridge=True)
        await transport(
            "exec_remote_start_wait", {"command": "ls"}, False, user="root", timeout=10
        )
        await close_bridges(proxy)

    [event] = [e for e in transcript().events if isinstance(e, SandboxEvent)]
    assert (event.action, event.cmd, event.result) == (
        "exec",
        f"{SANDBOX_CLI} bridge",
        0,
    )
    assert json.loads(event.input or "")["params"] == {"command": "ls"}
    assert event.options == {"user": "root", "timeout": 10}


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

    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, serve)
        proxy = SandboxEnvironmentProxy(sandbox)  # type: ignore[arg-type]
        result = await exec_remote_awaitable(
            proxy,
            ["ls"],
            5,
            ExecRemoteAwaitableOptions(server_wait=True, poll_timeout=10),
        )
        await close_bridges(proxy)

    assert (result.returncode, result.stdout, result.stderr) == (3, "out", "err")
    [bridge] = sandbox.bridges
    assert [r["method"] for r in bridge.requests] == ["exec_remote_start_wait"]
    assert sandbox.execs == []


@pytest.mark.usefixtures("bridge_on")
async def test_each_frame_limit_gets_its_own_bridge() -> None:
    async with anyio.create_task_group() as tg:
        sandbox = _BridgeSandbox(tg, _serve_forever)
        transport = _bridged_transport(sandbox)
        await _result(transport, "exec_remote_poll_wait")
        with override_max_exec_output_size(4096):
            await _result(transport, "exec_remote_poll_wait")
        await _result(transport, "exec_remote_poll_wait")
        await close_bridges(sandbox)  # type: ignore[arg-type]

    assert [cmd[-1] for cmd in sandbox.stdio_calls] == ["10485760", "4096"]
    assert [len(bridge.requests) for bridge in sandbox.bridges] == [2, 1]


@pytest.mark.usefixtures("bridge_on")
async def test_sample_cleanup_closes_the_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = anyio.Event()
    cleaned: list[dict[str, Any]] = []

    async def serve(bridge: _FakeBridge) -> None:
        await _serve_forever(bridge)
        closed.set()

    class _Provider:
        @classmethod
        async def sample_cleanup(cls, *args: Any) -> None:
            cleaned.append(args[2])

    monkeypatch.setattr(
        sandbox_context, "registry_find_sandboxenv", lambda type: _Provider
    )
    async with anyio.create_task_group() as tg:
        provider = _BridgeSandbox(tg, serve)
        proxy = SandboxEnvironmentProxy(provider)  # type: ignore[arg-type]
        transport = SandboxJSONRPCTransport(proxy, SANDBOX_CLI, bridge=True)
        await _result(transport, "exec_remote_start_wait")
        await sandbox_context.cleanup_sandbox_environments_sample(
            "fake", "task", None, {"default": proxy}, False
        )
        with anyio.fail_after(10):
            await closed.wait()

    assert cleaned == [{"default": provider}]
