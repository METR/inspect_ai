"""Many sandbox tools requests over one long-lived exec.

`SandboxJSONRPCTransport` normally starts the injected CLI in a new exec for each
request. When `INSPECT_EXEC_REMOTE_BRIDGE` is `1` or `true` and the sandbox provider
implements `SandboxEnvironment.exec_stdio`, exec_remote's server-wait requests go over
one `inspect-sandbox-tools bridge` exec per sandbox and tools user. The bridge
forwards each request line to the sandbox tools server, as `exec` does with its one
request, and writes each reply as a line.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

import anyio
from anyio.abc import ByteStream
from anyio.streams.buffered import BufferedByteStream

from inspect_ai._util.logger import warn_once

from ._cli import SANDBOX_CLI

if TYPE_CHECKING:
    from .environment import SandboxEnvironment

logger = logging.getLogger(__name__)

BRIDGE_ENV = "INSPECT_EXEC_REMOTE_BRIDGE"
"""Sends exec_remote's server-wait requests over a bridge when `1` or `true`."""

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


class BridgeUnavailable(Exception):
    """No bridge could send a request sent with `bridge_only`."""


class _Bridge:
    """One `bridge` exec: requests are written as lines, and replies matched by ID.

    No task reads the stream on its own. A caller waiting for its reply reads the
    next line when no other caller is reading, hands the reply to the request it
    answers, and lets the next waiting caller read. A reader that is cancelled
    loses no data, since a partial line stays buffered for the next reader. A
    reply for a request that stopped waiting (its caller timed out or was
    cancelled) is dropped; any other reply no request waits for drops the bridge.
    """

    def __init__(self, stream: ByteStream, max_frame_bytes: int) -> None:
        self._stream = BufferedByteStream(stream)
        self._max_frame_bytes = max_frame_bytes
        self._replies: dict[object, str | None] = {}
        self._abandoned: set[object] = set()  # stopped waiting before their reply
        self._reading = False
        self._read_scope: anyio.CancelScope | None = None
        self._turn = anyio.Event()
        self._send_lock = anyio.Lock(fast_acquire=True)  # free: no turn of the loop
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

    async def request(self, request_id: object, frame: bytes, deadline: float) -> str:
        """Send `frame` and wait for its reply; its write ends by `deadline`."""
        if self.dropped is not None:
            raise BridgeDropped(self.dropped)
        self._replies[request_id] = None
        try:
            await self._send(frame, deadline)
            while (reply := self._replies[request_id]) is None:
                if self.dropped is not None:
                    raise BridgeDropped(self.dropped)
                if self._reading:
                    await self._turn.wait()
                else:
                    await self._read_reply()
            return reply
        finally:
            if self._replies.pop(request_id) is None:
                self._abandoned.add(request_id)

    async def drop(self, reason: str) -> NoReturn:
        """Close the bridge, failing the requests waiting on it with `reason`."""
        await self.close(reason)
        raise BridgeDropped(self.dropped)

    async def close(self, reason: str) -> None:
        """Stop the bridge. Never raises: a stream that fails to close is dropped.

        The waiting requests fail with `reason` at once, whether or not closing
        the stream ends its reader's read.
        """
        if self.dropped is not None:
            return
        self.dropped = reason
        if self._read_scope is not None:
            self._read_scope.cancel()
        with anyio.move_on_after(_CLOSE_TIMEOUT, shield=True):
            try:
                await self._stream.aclose()
            except Exception as ex:
                logger.debug("Closing a sandbox tools bridge failed: %s", ex)

    async def _send(self, frame: bytes, deadline: float) -> None:
        async with self._send_lock:
            # Cancelling the caller does not cut a frame short, which would corrupt
            # the stream for every other request; only the request's deadline does.
            with anyio.CancelScope(shield=True, deadline=deadline) as scope:
                try:
                    await self._stream.send(frame)
                except Exception as ex:
                    await self.drop(f"writing to the bridge failed: {ex!r}")
            if scope.cancelled_caught:
                await self.close("a request could not be written in time")
                raise TimeoutError

    async def _read_reply(self) -> None:
        self._reading = True
        try:
            with anyio.CancelScope() as self._read_scope:
                line = await self._read_line()
            if self._read_scope.cancelled_caught:
                raise BridgeDropped(self.dropped)
            try:
                request_id = json.loads(line.decode("utf-8"))["id"]
                waiting = request_id in self._replies and not self._replies[request_id]
            except Exception:
                await self.drop(f"the bridge sent an invalid reply: {line[:200]!r}")
            if waiting:
                self._replies[request_id] = line.decode("utf-8")
            elif request_id in self._abandoned:
                self._abandoned.remove(request_id)
            else:
                await self.drop("the bridge sent a reply no request waits for")
        finally:
            self._read_scope = None
            self._reading = False
            turn, self._turn = self._turn, anyio.Event()
            turn.set()

    async def _read_line(self) -> bytes:
        try:
            line = await self._stream.receive_until(b"\n", self._max_frame_bytes)
        except anyio.DelimiterNotFound:
            await self.drop("the bridge sent a reply longer than the frame limit")
        except anyio.IncompleteRead:
            await self.drop("the bridge exited")
        except Exception as ex:
            await self.drop(f"reading from the bridge failed: {ex!r}")
        if len(line) >= self._max_frame_bytes:  # found past the limit in one read
            await self.drop("the bridge sent a reply longer than the frame limit")
        return line


@dataclass
class _Bridges:
    """A sandbox's bridges, by tools user and frame limit."""

    lock: anyio.Lock = field(default_factory=lambda: anyio.Lock(fast_acquire=True))
    by_key: dict[tuple[str | None, int], _Bridge | None] = field(default_factory=dict)
    """An open bridge, or None when the sandbox cannot run one."""
    closed: bool = False
    """Whether the sample has ended, after which no bridge opens."""


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
    Another error starting a bridge is raised, as the provider raised it, for the
    request that started it.

    A request lost with a dropped bridge is sent once more over a new bridge, or
    in its own exec (None) if no new bridge starts; a second loss raises
    `BridgeDropped`. A request that times out raises `SandboxTimeoutError` and
    closes the bridge, which may be wedged; the other requests on it are sent
    again. `timeout` covers the request, not starting the bridge
    (`OPEN_TIMEOUT`). The reply can be a `bridge_error`.
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
        deadline = math.inf if timeout is None else anyio.current_time() + timeout
        try:
            with anyio.fail_after(timeout):
                return await bridge.request(request_id, frame, deadline)
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
    raise AssertionError("unreachable")


def bridge_open(
    sandbox: SandboxEnvironment, user: str | None, max_frame_bytes: int
) -> bool:
    """Whether the sandbox has an open bridge for `user` and the frame limit."""
    bridges = sandbox._json_rpc_bridges
    bridge = bridges.by_key.get((user, max_frame_bytes)) if bridges else None
    return bridge is not None and bridge.dropped is None


def bridge_error(reply: str) -> str | None:
    """The error in a reply for a request the server never answered, if it is one."""
    if _ERROR_FIELD not in reply:
        return None
    error = json.loads(reply).get(_ERROR_FIELD)
    return error if isinstance(error, str) else None


async def close_bridges(sandbox: SandboxEnvironment) -> None:
    """Close the bridges a sandbox has open, when its sample ends. Never raises.

    No bridge opens after this, including one that was opening; later requests
    are sent in their own execs.
    """
    if sandbox._json_rpc_bridges is None:
        sandbox._json_rpc_bridges = _Bridges()
    bridges = sandbox._json_rpc_bridges
    bridges.closed = True
    for bridge in list(bridges.by_key.values()):
        if bridge is not None:
            await bridge.close("the sample ended")
    bridges.by_key.clear()


async def _bridge(
    sandbox: SandboxEnvironment, user: str | None, max_frame_bytes: int
) -> _Bridge | None:
    """The sandbox's open bridge for `user`, starting one if it has none.

    None when the sandbox cannot run one: its provider has no `exec_stdio`, or the
    bridge does not report ready within `OPEN_TIMEOUT`. That is remembered, so a
    sandbox whose bridge fails to start (including one started again after a
    drop) sends every later request in its own exec. Another error starting it
    is raised for the request that started it, and remembered the same way. The
    frame limit is the exec output limit, which a caller can lower for its
    requests, so each limit has its own bridge.
    """
    if sandbox._json_rpc_bridges is None:
        sandbox._json_rpc_bridges = _Bridges()
    bridges = sandbox._json_rpc_bridges
    key = (user, max_frame_bytes)
    async with bridges.lock:
        if bridges.closed:
            return None
        if key in bridges.by_key:
            bridge = bridges.by_key[key]
            if bridge is None or bridge.dropped is None:
                return bridge
        bridge = None
        try:
            with anyio.fail_after(OPEN_TIMEOUT):
                stream = await sandbox.exec_stdio(
                    [SANDBOX_CLI, "bridge", "--max-frame-bytes", str(max_frame_bytes)],
                    user=user,
                )
                bridge = _Bridge(stream, max_frame_bytes)
                await bridge.wait_ready()
        except NotImplementedError:
            bridges.by_key[key] = None
            return None
        except (TimeoutError, BridgeDropped) as ex:
            if bridge is not None:
                await bridge.close("the bridge did not report ready")
            warn_once(
                logger,
                "A sandbox's tools bridge did not report ready, so its requests "
                "are sent in separate execs (an older sandbox tools build has no "
                "bridge).",
            )
            logger.debug("The sandbox tools bridge did not report ready: %r", ex)
            bridges.by_key[key] = None
            return None
        except Exception:
            # raised for this request; the requests queued behind it use execs
            if bridge is not None:
                await bridge.close("starting the bridge failed")
            bridges.by_key[key] = None
            raise
        except BaseException:
            if bridge is not None:
                await bridge.close("starting the bridge was cancelled")
            raise
        if bridges.closed:  # the sample ended while it opened
            await bridge.close("the sample ended")
            return None
        bridges.by_key[key] = bridge
        return bridge
