"""SPIKE: JSON-RPC to the sandbox-tools server over the network instead of `exec`.

`SandboxJSONRPCTransport` runs `inspect-sandbox-tools exec` once per request: a new
provider exec session (for Kubernetes, an API-server round trip and websocket) and
a new process in the sandbox, every time. When the sandbox-tools server was started
with a TCP listener (see `tools_network_host()`), this transport POSTs the same
request to it directly over a kept-alive HTTP connection.

The wire format and the server's handlers are the ones the exec path already uses;
only the hop between host and server changes. A request that cannot connect falls
back to the exec transport (the server may have been killed and restarted without
the listener); any failure after the request may have reached the server is raised,
as the exec transport would, because replaying `exec_remote_start` could run a
command twice.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

# SPIKE: aiohttp, not httpx. One httpx pool serving hundreds of sandboxes spent a
# quarter of the runner's GIL time in httpcore's per-request scan of every pooled
# connection (`_assign_requests_to_connections`); aiohttp keys its pool by host and
# parses HTTP in C. aiohttp is not a declared inspect_ai dependency (Hawk's runner
# has it through s3fs).
import aiohttp

from inspect_ai._util._json_rpc import JSONRPCParamsType, create_json_rpc_request
from inspect_ai._util.working import sandbox_exec_work

from ._cli import SANDBOX_CLI

if TYPE_CHECKING:
    from ._json_rpc_transport import SandboxJSONRPCTransport
    from .environment import SandboxEnvironment

TCP_PORT_ENV = "INSPECT_SANDBOX_TOOLS_TCP_PORT"
"""Host-side opt-in: the port the sandbox-tools server listens on (unset = off)."""

_DEFAULT_TIMEOUT = 120

SERVER_MS_HEADER = "X-Sandbox-Tools-Ms"
"""SPIKE: the server's processing time for the request, in milliseconds."""

_session: aiohttp.ClientSession | None = None


def tools_tcp_port() -> int | None:
    port = os.environ.get(TCP_PORT_ENV)
    return int(port) if port else None


def _http_session() -> aiohttp.ClientSession:
    # One session for every sandbox in the process (created on first use, inside
    # the event loop); limit=0 so the connector never queues a request behind
    # another sandbox's.
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
    return _session


class HttpJSONRPCTransport:
    def __init__(
        self,
        sandbox: SandboxEnvironment,
        url: str,
        token: str,
        fallback: SandboxJSONRPCTransport,
    ) -> None:
        # `sandbox` mirrors SandboxJSONRPCTransport.sandbox: exec_remote's poll path
        # reads it to suppress transcript events for polls.
        self.sandbox = sandbox
        self._url = url
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._fallback = fallback

    async def __call__(
        self,
        method: str,
        params: JSONRPCParamsType,
        is_notification: bool,
        **transport_extra_args: Any,
    ) -> str:
        request = create_json_rpc_request(method, params, is_notification)
        timeout = transport_extra_args.get("timeout") or _DEFAULT_TIMEOUT
        started = datetime.now(timezone.utc)
        try:
            with sandbox_exec_work():
                async with _http_session().post(
                    self._url,
                    data=request,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    status, text = response.status, await response.text()
                    server_ms = response.headers.get(SERVER_MS_HEADER)
        except aiohttp.ClientConnectorError:
            # nothing reached the server, so the exec path can safely take it
            return await self._fallback(
                method, params, is_notification, **transport_extra_args
            )
        except asyncio.TimeoutError as ex:
            raise TimeoutError(
                f"sandbox-tools request {method} timed out after {timeout}s"
            ) from ex
        except aiohttp.ClientError as ex:
            raise RuntimeError(f"sandbox-tools request {method} failed: {ex}") from ex
        if status != 200:
            raise RuntimeError(
                f"sandbox-tools request {method} failed: HTTP {status} {text[:200]}"
            )
        self._record_event(started, request, text, server_ms)
        return text

    def _record_event(
        self, started: datetime, request: str, response: str, server_ms: str | None
    ) -> None:
        # Keep the transcript as the exec path writes it (one sandbox event for the
        # requests exec_remote does not mark as polls, output left out where the
        # exec path leaves it out), so logs and timing analyses stay comparable.
        from .events import SandboxEnvironmentProxy, _event_output_off, content_display

        proxy = self.sandbox
        if (
            not isinstance(proxy, SandboxEnvironmentProxy)
            or not proxy._events_enabled()
        ):
            return
        from inspect_ai.event._sandbox import SandboxEvent
        from inspect_ai.log._transcript import transcript

        transcript()._event(
            SandboxEvent(
                timestamp=started,
                action="exec",
                # SPIKE: the server's own time is in the command text too, because
                # the warehouse keeps `cmd` but not `options`
                cmd=f"{SANDBOX_CLI} exec [http server_ms={server_ms}]",
                input=content_display(request),
                options={"transport": "http", "server_ms": server_ms},
                result=0,
                output=None
                if proxy in _event_output_off.get()
                else content_display(response),
                completed=datetime.now(timezone.utc),
            )
        )
