"""Many sandbox tools requests over one long-lived exec.

`SandboxJSONRPCTransport` normally starts the injected CLI in a new exec for each
request. When `INSPECT_EXEC_REMOTE_BRIDGE` is set and the sandbox provider
implements `SandboxEnvironment.exec_stdio`, exec_remote's requests instead go over
one `inspect-sandbox-tools bridge` exec per sandbox and tools user. The bridge
forwards each request line to the sandbox tools server, as `exec` does with its one
request, and writes each reply as a line.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

import anyio
from anyio.abc import ByteStream

from ._cli import SANDBOX_CLI

if TYPE_CHECKING:
    from .environment import SandboxEnvironment

logger = logging.getLogger(__name__)

BRIDGE_ENV = "INSPECT_EXEC_REMOTE_BRIDGE"
"""Sends exec_remote's requests over a bridge when `1` or `true`."""

BRIDGE_METHODS = frozenset(
    {
        "exec_remote_start_wait",
        "exec_remote_poll_wait",
        "exec_remote_kill_wait",
        "__inspect_json_rpc_response_chunk__",
    }
)
"""Requests sent over a bridge. Each names a request ID or a spilled reply, so the
server can receive one twice without effect, and one lost with a dropped bridge is
sent again."""

OPEN_TIMEOUT = 60.0
"""Seconds a new bridge has to report that it is ready."""

_CLOSE_TIMEOUT = 5.0

# Also defined in inspect_sandbox_tools._cli.main — keep in sync.
_READY = {
    "jsonrpc": "2.0",
    "method": "__inspect_bridge_ready__",
    "params": {"version": 1},
}
_ERROR_FIELD = "__inspect_bridge_error__"


def bridge_enabled() -> bool:
    return os.environ.get(BRIDGE_ENV, "").strip().lower() in ("1", "true")


class BridgeDropped(RuntimeError):
    """The bridge ended, broke or misbehaved before a request's reply arrived."""


class _Bridge:
    """One `bridge` exec: requests are written as lines, and replies matched by ID.

    No task reads the stream on its own. A caller waiting for its reply reads the
    next line when no other caller is reading, hands the reply to the request it
    answers, and lets the next waiting caller read. A reader that is cancelled
    loses no data, since a partial line stays buffered for the next reader, and
    a reply for a request nobody waits for (its caller timed out or was
    cancelled) is dropped.
    """

    def __init__(self, stream: ByteStream, max_frame_bytes: int) -> None:
        self._stream = stream
        self._max_frame_bytes = max_frame_bytes
        self._replies: dict[object, str | None] = {}
        self._buffer = bytearray()
        self._scanned = 0  # bytes at the start of the buffer that hold no newline
        self._reading = False
        self._turn = anyio.Event()
        self._send_lock = anyio.Lock()
        self.dropped: str | None = None
        """Why the bridge can take no more requests, once it cannot."""

    async def wait_ready(self) -> None:
        line = await self._read_line()
        try:
            ready = json.loads(line)
        except ValueError:
            ready = None
        if ready != _READY:
            await self.drop(f"the bridge did not start: {line[:200]!r}")

    async def request(self, request_id: object, frame: bytes) -> str:
        if self.dropped is not None:
            raise BridgeDropped(self.dropped)
        self._replies[request_id] = None
        try:
            await self._send(frame)
            while (reply := self._replies[request_id]) is None:
                if self.dropped is not None:
                    raise BridgeDropped(self.dropped)
                if self._reading:
                    await self._turn.wait()
                else:
                    await self._read_reply()
            return reply
        finally:
            del self._replies[request_id]

    async def drop(self, reason: str) -> NoReturn:
        """Close the bridge, failing the requests waiting on it with `reason`."""
        await self.close(reason)
        raise BridgeDropped(self.dropped)

    async def close(self, reason: str) -> None:
        if self.dropped is not None:
            return
        self.dropped = reason
        with anyio.move_on_after(_CLOSE_TIMEOUT, shield=True):
            await self._stream.aclose()

    async def _send(self, frame: bytes) -> None:
        async with self._send_lock:
            # A frame cut short would corrupt every later one, so finish writing it
            # unless the request's own deadline passes.
            deadline = anyio.current_effective_deadline()
            with anyio.CancelScope(shield=True, deadline=deadline) as scope:
                try:
                    await self._stream.send(frame)
                except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                    await self.drop("the bridge's stdin closed")
            if scope.cancelled_caught:
                await self.drop("a request could not be written in time")

    async def _read_reply(self) -> None:
        self._reading = True
        try:
            line = await self._read_line()
            try:
                request_id = json.loads(line.decode("utf-8"))["id"]
                waiting = request_id in self._replies and not self._replies[request_id]
            except (ValueError, LookupError, TypeError):
                await self.drop(f"the bridge sent an invalid reply: {line[:200]!r}")
            if waiting:
                self._replies[request_id] = line.decode("utf-8")
        finally:
            self._reading = False
            turn, self._turn = self._turn, anyio.Event()
            turn.set()

    async def _read_line(self) -> bytes:
        while (end := self._buffer.find(b"\n", self._scanned)) < 0:
            self._scanned = len(self._buffer)
            if self._scanned >= self._max_frame_bytes:
                await self.drop("the bridge sent a reply longer than the frame limit")
            try:
                data = await self._stream.receive()
            except anyio.EndOfStream:
                await self.drop("the bridge exited")
            except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                await self.drop("the bridge's stdout closed")
            self._buffer += data
        line = bytes(self._buffer[:end])
        del self._buffer[: end + 1]
        self._scanned = 0
        return line


@dataclass
class _Bridges:
    """A sandbox's bridges, by tools user and frame limit."""

    lock: anyio.Lock = field(default_factory=anyio.Lock)
    by_key: dict[tuple[str | None, int], _Bridge | None] = field(default_factory=dict)
    """An open bridge, or None when the sandbox cannot run one."""


async def bridge_request(
    sandbox: SandboxEnvironment,
    user: str | None,
    request: str,
    description: str,
    timeout: float | None,
    max_frame_bytes: int,
) -> str | None:
    """Send a request over the sandbox's bridge for `user`, starting it if needed.

    Returns None, so the request is sent in its own exec, when there is no bridge:
    the provider does not implement `exec_stdio`, the bridge did not start (an
    older sandbox tools build, for one), or the request does not fit in a frame.

    A request lost with a dropped bridge is sent once more over a new bridge, or
    in its own exec (None) if no new bridge starts; a second loss raises
    `BridgeDropped`. A request that times out raises `SandboxTimeoutError` and
    closes the bridge, which may be wedged; the other requests on it are sent
    again. `timeout` covers the request, not starting the bridge
    (`OPEN_TIMEOUT`). A request the server never answered raises `RuntimeError`,
    as a failed exec of the CLI does.
    """
    from .events import SandboxTimeoutError

    frame = request.encode("utf-8") + b"\n"
    if len(frame) > max_frame_bytes:
        return None
    request_id = json.loads(request)["id"]
    for attempt in (1, 2):
        bridge = await _bridge(sandbox, user, max_frame_bytes)
        if bridge is None:
            return None
        try:
            with anyio.fail_after(timeout):
                reply = await bridge.request(request_id, frame)
        except TimeoutError as ex:
            await bridge.close(f"{description} timed out")
            raise SandboxTimeoutError(
                f"Timed out after {timeout}s waiting for the sandbox tools bridge "
                f"to answer {description}"
            ) from ex
        except BridgeDropped as ex:
            if attempt == 2:
                raise
            logger.info("Resending %s: %s", description, ex)
            continue
        if _ERROR_FIELD in (payload := json.loads(reply)):
            raise RuntimeError(
                f"Sandbox.exec failure executing {description}: {payload[_ERROR_FIELD]}"
            )
        return reply
    raise AssertionError("unreachable")


async def close_bridges(sandbox: SandboxEnvironment) -> None:
    """Close the bridges a sandbox has open (at sample cleanup)."""
    bridges = sandbox._json_rpc_bridges
    if bridges is None:
        return
    async with bridges.lock:
        for bridge in bridges.by_key.values():
            if bridge is not None:
                await bridge.close("the sandbox was cleaned up")
        bridges.by_key.clear()


async def _bridge(
    sandbox: SandboxEnvironment, user: str | None, max_frame_bytes: int
) -> _Bridge | None:
    """The sandbox's open bridge for `user`, starting one if it has none.

    None when the sandbox cannot run one. That is remembered, so a sandbox whose
    bridge fails to start (including one started again after a drop) sends every
    later request in its own exec. The frame limit is the exec output limit,
    which a caller can lower for its requests, so each limit has its own bridge.
    """
    if sandbox._json_rpc_bridges is None:
        sandbox._json_rpc_bridges = _Bridges()
    bridges = sandbox._json_rpc_bridges
    key = (user, max_frame_bytes)
    async with bridges.lock:
        if key in bridges.by_key:
            bridge = bridges.by_key[key]
            if bridge is None or bridge.dropped is None:
                return bridge
        try:
            stream = await sandbox.exec_stdio(
                [SANDBOX_CLI, "bridge", "--max-frame-bytes", str(max_frame_bytes)],
                user=user,
            )
        except NotImplementedError:
            bridges.by_key[key] = None
            return None
        except Exception as ex:
            logger.warning(
                "Sending sandbox tools requests in separate execs: the bridge "
                "could not be started (%s)",
                ex,
            )
            bridges.by_key[key] = None
            return None
        bridge = _Bridge(stream, max_frame_bytes)
        try:
            with anyio.fail_after(OPEN_TIMEOUT):
                await bridge.wait_ready()
        except (TimeoutError, BridgeDropped) as ex:
            await bridge.close("the bridge did not report ready")
            logger.warning(
                "Sending sandbox tools requests in separate execs: the bridge did "
                "not report ready (%s)",
                ex,
            )
            bridges.by_key[key] = None
            return None
        except BaseException:
            await bridge.close("starting the bridge was cancelled")
            raise
        bridges.by_key[key] = bridge
        return bridge
