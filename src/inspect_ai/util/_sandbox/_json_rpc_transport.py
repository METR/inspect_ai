"""JSON-RPC transport implementation for sandbox environments."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from inspect_ai._util._json_rpc import (
    JSONRPCParamsType,
    JSONRPCTransport,
    create_json_rpc_request,
    rpc_call_description,
)

if TYPE_CHECKING:
    from .environment import SandboxEnvironment

from .._subprocess import ExecResult
from ._cli import SANDBOX_CLI, SANDBOX_TOOLS_DIR, TOOLS_DIR_REFUSED, tools_cli_command
from ._json_rpc_bridge import (
    BRIDGE_METHODS,
    BridgeUnavailable,
    bridge_enabled,
    bridge_error,
    bridge_open,
    bridge_request,
)
from .limits import OutputLimitExceededError, SandboxEnvironmentLimits


class SandboxReplacedError(RuntimeError):
    """A launch of the sandbox tools was refused after their install check passed.

    The tools tree went away since, so the sandbox was most likely replaced and
    lost the sample's state. Not a tool error: the sample fails.
    """


def sandbox_replaced_error(failed: str, reason: str = "") -> SandboxReplacedError:
    """The error for a launch of the tools CLI that printed `TOOLS_DIR_REFUSED`."""
    return SandboxReplacedError(
        "The sandbox was most likely replaced (its container restarted, or its"
        " VM died) and lost the sample's state: the sandbox tools directory"
        f" installed earlier ({SANDBOX_TOOLS_DIR}) is missing or no longer the"
        " one installed, unless something in the sandbox removed or changed it."
        f" The tools are not installed again. Failed call: {failed}"
        + (f". The check reported: {reason}" if reason else "")
    )


_JSON_RPC_RESPONSE_CHUNK_METHOD = "__inspect_json_rpc_response_chunk__"
_JSON_RPC_RESPONSE_CHUNK_FIELD = "__inspect_json_rpc_response_chunk__"
_JSON_RPC_RESPONSE_CHUNK_VERSION = 1
_JSON_RPC_RESPONSE_MAX_BYTES_ENV = "INSPECT_SANDBOX_JSON_RPC_RESPONSE_MAX_BYTES"
_VALID_CHUNK_HANDLE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class _ResponseChunk:
    handle: str
    offset: int
    next_offset: int
    total_size: int
    done: bool
    data: bytes


class SandboxJSONRPCTransport(JSONRPCTransport):
    """A transport that uses a sandbox for RPC communication.

    This class implements the JSONRPCTransport protocol. The timeout and user
    parameters are passed via transport_extra_args in the __call__ method.
    """

    def __init__(
        self,
        sandbox: SandboxEnvironment,
        cli: str,
        *,
        bridge: bool = False,
    ):
        """Initialize a new SandboxJSONRPCTransport.

        Args:
            sandbox: The sandbox environment to use.
            cli: The path to the cli available in the sandbox.
            bridge: Send the requests in `BRIDGE_METHODS` over the sandbox's
                bridge when `INSPECT_EXEC_REMOTE_BRIDGE` is `1` or `true` (see
                `_json_rpc_bridge`).
        """
        self.sandbox = sandbox
        self.cli = cli
        self._response_chunking = cli == SANDBOX_CLI
        self._bridge = bridge

    def bridged(self, method: str, user: str | None) -> bool:
        """Whether a request would go over an open bridge, so it needs no exec.

        Send it with `bridge_only`, since the bridge can close (or the request
        not fit in a frame) before it is sent.
        """
        return (
            self._bridge
            and method in BRIDGE_METHODS
            and bridge_enabled()
            and bridge_open(
                self.sandbox, user, SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
            )
        )

    async def __call__(
        self,
        method: str,
        params: JSONRPCParamsType,
        is_notification: bool,
        **transport_extra_args: Any,
    ) -> str:
        """Execute an RPC request using the sandbox transport.

        Args:
            method: The JSON-RPC method to call.
            params: The parameters for the JSON-RPC method.
            is_notification: Whether this is a notification (no response expected).
            **transport_extra_args: Additional parameters including timeout and
                user, and `bridge_only`, which sends the request only over a bridge.

        Returns:
            The response from the RPC call.

        Raises:
            RuntimeError: If the sandbox execution fails.
            BridgeUnavailable: If no bridge sent a `bridge_only` request.
        """
        bridge_only = transport_extra_args.pop("bridge_only", False)
        request = create_json_rpc_request(method, params, is_notification)
        max_response_bytes = (
            SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
            if self._response_chunking
            else None
        )
        response = await self._sandbox_exec(
            method,
            request,
            rpc_call_description(method, params),
            transport_extra_args,
            max_response_bytes,
            bridge_only,
        )
        if max_response_bytes is None:
            return response
        return await self._complete_chunked_response(
            response, transport_extra_args, max_response_bytes
        )

    async def _sandbox_exec(
        self,
        method: str,
        request: str,
        description: str,
        transport_extra_args: dict[str, Any],
        max_response_bytes: int | None,
        bridge_only: bool = False,
    ) -> str:
        if (
            self._bridge
            and max_response_bytes is not None
            and method in BRIDGE_METHODS
            and bridge_enabled()
        ):
            started = datetime.now(timezone.utc)
            reply = await bridge_request(
                self.sandbox,
                transport_extra_args.get("user", None),
                request,
                description,
                transport_extra_args.get("timeout", None),
                max_response_bytes,
            )
            if reply is not None:
                error = bridge_error(reply)
                self._record_bridged(
                    started, request, reply, error, transport_extra_args
                )
                if error is not None:
                    raise RuntimeError(
                        f"Sandbox.exec failure executing {description}: {error}"
                    )
                return reply
        if bridge_only:
            raise BridgeUnavailable(f"No sandbox tools bridge sent {description}")
        env = (
            {_JSON_RPC_RESPONSE_MAX_BYTES_ENV: str(max_response_bytes)}
            if max_response_bytes is not None
            else None
        )
        tools_cli = self.cli == SANDBOX_CLI
        user = transport_extra_args.get("user", None)
        exec_result = await self.sandbox.exec(
            tools_cli_command("exec", user=user) if tools_cli else [self.cli, "exec"],
            input=request,
            env=env,
            timeout=transport_extra_args.get("timeout", None),
            timeout_retry=transport_extra_args.get("timeout_retry", True),
            user=user,
            concurrency=transport_extra_args.get("concurrency", True),
        )
        if tools_cli and exec_result.stdout.strip() == TOOLS_DIR_REFUSED:
            raise sandbox_replaced_error(description, exec_result.stderr.strip())

        if not exec_result.success:
            # Prefer stderr, but fall back to stdout — some failures
            # (e.g. MCP server crash, entrypoint error) only surface in
            # stdout because the sandbox CLI wrote its diagnostic there.
            error_detail = (
                exec_result.stderr or exec_result.stdout or "(no output captured)"
            )
            raise RuntimeError(
                f"Sandbox.exec failure executing {description}: {error_detail}"
            )
        return exec_result.stdout

    async def _complete_chunked_response(
        self,
        response: str,
        transport_extra_args: dict[str, Any],
        max_response_bytes: int,
    ) -> str:
        chunk = _parse_response_chunk(response)
        if chunk is None:
            return response

        response_bytes = bytearray()
        handle = chunk.handle
        total_size = chunk.total_size
        expected_offset = 0
        whole = False
        try:
            # The reply is held whole in the host's memory, at whatever size the
            # sandbox declares, so it gets the limit on reading a file into memory.
            if total_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
                raise OutputLimitExceededError(
                    SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR, None
                )
            while True:
                _validate_response_chunk(
                    chunk,
                    expected_handle=handle,
                    expected_offset=expected_offset,
                    expected_total_size=total_size,
                    max_response_bytes=max_response_bytes,
                )
                response_bytes.extend(chunk.data)
                if chunk.done:
                    if len(response_bytes) != total_size:
                        raise RuntimeError(
                            "Chunked JSON-RPC response size did not match metadata"
                        )
                    try:
                        text = response_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        raise RuntimeError(
                            "Chunked JSON-RPC response was not valid UTF-8"
                        ) from None
                    whole = True
                    return text

                expected_offset = chunk.next_offset
                next_response = await self._sandbox_exec(
                    _JSON_RPC_RESPONSE_CHUNK_METHOD,
                    create_json_rpc_request(
                        _JSON_RPC_RESPONSE_CHUNK_METHOD,
                        {"handle": handle, "offset": expected_offset},
                        False,
                    ),
                    "chunked JSON-RPC response continuation",
                    transport_extra_args,
                    max_response_bytes,
                )
                next_chunk = _parse_response_chunk(next_response, require_chunk=True)
                assert next_chunk is not None
                chunk = next_chunk
        finally:
            await self._release_chunked_response(
                handle, transport_extra_args, max_response_bytes, whole
            )

    async def _release_chunked_response(
        self,
        handle: str,
        transport_extra_args: dict[str, Any],
        max_response_bytes: int,
        whole: bool,
    ) -> None:
        """Release a chunked reply's files on the server.

        A refused release fails a reply that came in `whole`; it does not replace an
        error already on its way (a cancellation, say).
        """
        cleanup_args = {
            **transport_extra_args,
            "timeout": 5,
            "timeout_retry": False,
        }
        try:
            await self._sandbox_exec(
                _JSON_RPC_RESPONSE_CHUNK_METHOD,
                create_json_rpc_request(
                    _JSON_RPC_RESPONSE_CHUNK_METHOD,
                    {"handle": handle, "release": True},
                    False,
                ),
                "chunked JSON-RPC response cleanup",
                cleanup_args,
                max_response_bytes,
            )
        except SandboxReplacedError:
            if whole:
                raise
        except Exception:
            pass  # best effort: the server drops stale chunks itself

    def _record_bridged(
        self,
        started: datetime,
        request: str,
        reply: str,
        error: str | None,
        transport_extra_args: dict[str, Any],
    ) -> None:
        """Record the exec event its own exec would, with the command `<cli> bridge`."""
        from .events import SandboxEnvironmentProxy

        if isinstance(self.sandbox, SandboxEnvironmentProxy):
            self.sandbox.record_exec(
                started,
                [self.cli, "bridge"],
                request,
                ExecResult(True, 0, reply, "")
                if error is None
                else ExecResult(False, 1, "", error),
                user=transport_extra_args.get("user", None),
                timeout=transport_extra_args.get("timeout", None),
                timeout_retry=transport_extra_args.get("timeout_retry", True),
            )


def _parse_response_chunk(
    response: str, *, require_chunk: bool = False
) -> _ResponseChunk | None:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        if require_chunk:
            raise RuntimeError("Chunk continuation did not return valid JSON")
        return None

    if not isinstance(payload, dict):
        if require_chunk:
            raise RuntimeError("Chunk continuation did not return a JSON object")
        return None
    if _JSON_RPC_RESPONSE_CHUNK_FIELD not in payload:
        if require_chunk:
            if "error" in payload:
                raise RuntimeError(
                    f"Chunked JSON-RPC response fetch failed: {payload['error']}"
                )
            raise RuntimeError("Chunk continuation did not return chunk metadata")
        return None

    if payload.get("jsonrpc") != "2.0":
        raise RuntimeError("Chunked JSON-RPC response had an invalid protocol version")
    metadata = payload[_JSON_RPC_RESPONSE_CHUNK_FIELD]
    if not isinstance(metadata, dict):
        raise RuntimeError("Chunked JSON-RPC response metadata was not an object")
    if metadata.get("version") != _JSON_RPC_RESPONSE_CHUNK_VERSION:
        raise RuntimeError("Unsupported chunked JSON-RPC response version")

    handle = metadata.get("handle")
    offset = _chunk_int(metadata, "offset")
    next_offset = _chunk_int(metadata, "next_offset")
    total_size = _chunk_int(metadata, "total_size")
    done = metadata.get("done")
    encoded = metadata.get("chunk")
    if not isinstance(handle, str) or not _VALID_CHUNK_HANDLE.fullmatch(handle):
        raise RuntimeError("Invalid chunked JSON-RPC response handle")
    if not isinstance(done, bool):
        raise RuntimeError("Invalid chunked JSON-RPC response completion flag")
    if not isinstance(encoded, str):
        raise RuntimeError("Invalid chunked JSON-RPC response payload")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError("Invalid base64 in chunked JSON-RPC response") from None

    return _ResponseChunk(
        handle=handle,
        offset=offset,
        next_offset=next_offset,
        total_size=total_size,
        done=done,
        data=data,
    )


def _chunk_int(metadata: dict[str, Any], field: str) -> int:
    value = metadata.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"Invalid chunked JSON-RPC response {field}")
    return value


def _validate_response_chunk(
    chunk: _ResponseChunk,
    *,
    expected_handle: str,
    expected_offset: int,
    expected_total_size: int,
    max_response_bytes: int,
) -> None:
    if chunk.handle != expected_handle:
        raise RuntimeError("Chunked JSON-RPC response handle changed")
    if chunk.offset != expected_offset:
        raise RuntimeError("Chunked JSON-RPC response chunks arrived out of order")
    if chunk.total_size != expected_total_size or chunk.total_size <= 0:
        raise RuntimeError("Chunked JSON-RPC response total size changed")
    if not chunk.data or chunk.next_offset != chunk.offset + len(chunk.data):
        raise RuntimeError("Chunked JSON-RPC response did not make valid progress")
    if chunk.next_offset > chunk.total_size:
        raise RuntimeError("Chunked JSON-RPC response exceeded its declared size")
    if chunk.done != (chunk.next_offset == chunk.total_size):
        raise RuntimeError("Chunked JSON-RPC response completion metadata was invalid")
    # Each piece costs a sandbox exec and an event, so a reply may not come in many
    # more than the tools make: they fill each frame, up to 512 KiB, after an
    # envelope of under 320 bytes, with base64 (3 bytes in 4). A frame of 320 bytes
    # or less leaves the tools themselves only a few bytes a piece.
    if not chunk.done and len(chunk.data) < min(
        64 * 1024, max(max_response_bytes - 320, 0) * 3 // 8
    ):
        raise RuntimeError("Chunked JSON-RPC response came in too many pieces")
