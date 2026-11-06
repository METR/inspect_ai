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

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from inspect_ai._util._json_rpc import JSONRPCParamsType, create_json_rpc_request
from inspect_ai._util.working import sandbox_exec_work

from ._cli import SANDBOX_CLI

if TYPE_CHECKING:
    from ._json_rpc_transport import SandboxJSONRPCTransport
    from .environment import SandboxEnvironment

TCP_PORT_ENV = "INSPECT_SANDBOX_TOOLS_TCP_PORT"
"""Host-side opt-in: the port the sandbox-tools server listens on (unset = off)."""

_DEFAULT_TIMEOUT = 120

_client: httpx.AsyncClient | None = None


def tools_tcp_port() -> int | None:
    port = os.environ.get(TCP_PORT_ENV)
    return int(port) if port else None


def _http_client() -> httpx.AsyncClient:
    # One pool for every sandbox in the process: each sandbox keeps its own
    # connections alive, so the pool must not cap them at httpx's default of 100.
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=4096),
            timeout=_DEFAULT_TIMEOUT,
        )
    return _client


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
                response = await _http_client().post(
                    self._url, content=request, headers=self._headers, timeout=timeout
                )
        except httpx.ConnectError:
            # nothing reached the server, so the exec path can safely take it
            return await self._fallback(
                method, params, is_notification, **transport_extra_args
            )
        except httpx.TimeoutException as ex:
            raise TimeoutError(
                f"sandbox-tools request {method} timed out after {timeout}s"
            ) from ex
        except httpx.HTTPError as ex:
            raise RuntimeError(f"sandbox-tools request {method} failed: {ex}") from ex
        if response.status_code != 200:
            raise RuntimeError(
                f"sandbox-tools request {method} failed: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        self._record_event(started, request, response.text)
        return response.text

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
