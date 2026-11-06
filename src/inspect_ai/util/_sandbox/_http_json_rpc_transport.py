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
        self._record_event(started, request, text)
        return text

    def _record_event(self, started: datetime, request: str, response: str) -> None:
        # Keep the transcript as the exec path writes it (one sandbox event for the
        # requests exec_remote does not mark as polls), so logs and timing analyses
        # stay comparable across transports.
        if not getattr(self.sandbox, "_events", False):
            return
        from inspect_ai.event._sandbox import SandboxEvent
        from inspect_ai.log._transcript import transcript

        from .events import content_display

        transcript()._event(
            SandboxEvent(
                timestamp=started,
                action="exec",
                cmd=f"{SANDBOX_CLI} exec",
                input=content_display(request),
                options={"transport": "http"},
                result=0,
                output=content_display(response),
                completed=datetime.now(timezone.utc),
            )
        )
