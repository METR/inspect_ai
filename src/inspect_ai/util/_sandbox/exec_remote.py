"""ExecRemote - Asynchronous command execution with streaming output.

This module provides the host-side implementation for exec_remote, enabling
long-running commands in sandbox environments with streaming output.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, TypeVar, Union, cast
from uuid import uuid4

import anyio
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from inspect_ai._util._json_rpc import (
    GenericJSONRPCErrorMapper,
    JSONRPCErrorMapper,
    JSONRPCMethodNotFoundError,
    JSONRPCParamsType,
    exec_model_request,
)
from inspect_ai._util.cpu import effective_cpu_count
from inspect_ai.util._concurrency import (
    ConcurrencySemaphore,
    get_or_create_semaphore,
)

from ._cli import SANDBOX_CLI, tools_user_param
from ._json_rpc_transport import SandboxJSONRPCTransport

if TYPE_CHECKING:
    from .._subprocess import ExecResult
    from .environment import SandboxEnvironment

logger = logging.getLogger(__name__)


# ============================================================================
# Event Types
# ============================================================================


@dataclass
class ExecStdout:
    """A chunk of stdout data from the running process."""

    type: ClassVar[str] = "stdout"
    """Event type discriminator."""

    data: str
    """The stdout data."""


@dataclass
class ExecStderr:
    """A chunk of stderr data from the running process."""

    type: ClassVar[str] = "stderr"
    """Event type discriminator."""

    data: str
    """The stderr data."""


@dataclass
class ExecCompleted:
    """Process completed (successfully or with error)."""

    type: ClassVar[str] = "completed"
    """Event type discriminator."""

    exit_code: int
    """The process exit code (0 = success)"""

    @property
    def success(self) -> bool:
        """True if the process exited successfully (exit code 0)."""
        return self.exit_code == 0


ExecOutput = Union[ExecStdout, ExecStderr, ExecCompleted]
"""Union type for all events that can be yielded by ExecRemoteProcess.events."""


# ============================================================================
# Options
# ============================================================================


@dataclass
class ExecRemoteCommonOptions:
    """Common options for exec_remote() command execution.

    This base class contains options valid for both streaming and awaitable
    modes. Use ExecRemoteStreamingOptions or ExecRemoteAwaitableOptions for
    mode-specific options.
    """

    input: str | bytes | None = None
    """Standard input to send to the command"""

    cwd: str | None = None
    """Working directory for command execution"""

    env: dict[str, str] | None = None
    """Additional environment variables"""

    user: str | None = None
    """User to run the command as"""

    poll_interval: float | None = None
    """Interval between poll requests in seconds (at least 5).

    When this is not set, the first polls that find the command still running
    with no output are followed by shorter waits of 0.25, 0.5, 1 and 2 seconds
    (`POLL_RAMP_SECONDS`), so a short command is not left waiting a whole
    interval. These polls are added to the ones the sandbox's default interval
    makes and are skipped when they would not reply before the next of those.
    When it is set, every such poll is followed by this interval.

    A poll that returns output is followed by another at once, without using up
    the ramp. In awaitable mode (`stream=False`), where the output is read only
    when the command ends, that next poll only checks whether the command has
    ended: while polls in a row keep returning output, each later one is
    followed by `OUTPUT_REPOLL_SECONDS` (whatever this is set to), unless a
    reply is full (`MAX_POLL_OUTPUT_BYTES` of stdout or stderr)."""

    poll_timeout: float | None = None
    """Timeout for individual RPC poll requests in seconds. Defaults to 120 seconds.

    A poll that times out raises `TimeoutError` unless `poll_timeout_recovery` is
    set."""

    poll_timeout_retry: bool | None = None
    """Retry individual RPC requests (start, poll, stdin, kill) when they time out.
    Requests will be retried up to twice, with a timeout of no greater
    than 60 seconds for the first retry and 30 for the second. `False` cannot be
    combined with `poll_timeout_recovery`."""

    concurrency: bool = True
    """For sandboxes that run locally, request that the `concurrency()`
    function be used to throttle concurrent subprocesses. Requests held by
    `server_wait` are throttled by their own limit instead (see `server_wait`)."""

    poll_timeout_recovery: float | None = field(default=None, kw_only=True)
    """Seconds to keep re-issuing a poll after it times out. Defaults to `None`.

    Unset, a poll that times out raises `TimeoutError`. When set, a poll that
    times out is re-issued 5 seconds later, until the sandbox answers or a poll
    times out this many seconds or more after the first timeout, when that
    timeout is raised. The deadline is checked only when a poll times out, and a
    re-issued poll waits its full `poll_timeout` (plus any `poll_timeout_retry`),
    so recovery can run past this value by the 5-second pause plus one re-issued
    poll, including that poll's retries. While the command is running, a
    re-issued poll replays any output the lost response carried; if the command
    ended during the stall, its exit status is lost and `RuntimeError` is raised
    (with `server_wait`, the server keeps the final reply for a limited time).
    Only polls are re-issued, and none once `kill()` has been called, even if the
    kill request itself timed out; the error the last poll got is raised instead.
    A caller's `timeout` (awaitable mode) or cancellation still ends the wait at
    once. Raises `ValueError` with `poll_timeout_retry=False`, which asks for
    timed-out requests not to be retried.
    """

    def __post_init__(self) -> None:
        if self.poll_timeout_recovery is not None and self.poll_timeout_retry is False:
            raise ValueError(
                "poll_timeout_recovery re-issues timed-out polls, which "
                "poll_timeout_retry=False asks not to retry; set only one of them."
            )


@dataclass
class ExecRemoteStreamingOptions(ExecRemoteCommonOptions):
    """Options for exec_remote() in streaming mode (stream=True)."""

    stdin_open: bool = False
    """If True, keep stdin open after writing initial input, enabling write_stdin()
    and close_stdin() on the returned ExecRemoteProcess. If False (default), stdin
    is closed immediately after writing initial input (or not opened at all)"""


@dataclass
class ExecRemoteAwaitableOptions(ExecRemoteCommonOptions):
    """Options for exec_remote() in awaitable mode (stream=False)."""

    timeout: float | None = None
    """Maximum execution time in seconds. On timeout, the process is killed and
    TimeoutError is raised. With `server_wait`, this includes starting the command."""

    server_wait: bool = field(
        default_factory=lambda: _server_wait_default(), kw_only=True
    )
    """Wait for output in the sandbox instead of polling. Defaults to `False`, or
    to `True` when `INSPECT_EXEC_REMOTE_SERVER_WAIT` is `1` or `true`.

    When `True` and the sandbox tools server supports it, the server holds each
    request until the command exits or a full reply of output is ready, for at
    most 5 seconds and a quarter of `poll_timeout` (leaving the rest for the
    transport), so a quick command completes in one request. Requests are
    held only for the first 60 seconds of a command; after that, it is polled
    as usual. Held requests do not take the subprocess or Docker
    CLI slots that ordinary commands need; with `concurrency`, at most
    `INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY` of them (default: twice the
    available CPUs, at least 4) are in flight at once. A request never waits
    for that limit: when it is reached, the request is sent without a hold, as
    an ordinary command, and a command still running with no output is polled
    again as usual.

    A start that fails without an answer from the server (a lost connection, or
    a timeout unless `poll_timeout_retry` is False) is resent once with the same
    request ID, within 60 seconds of the first attempt. The server then reports
    the command it already launched, or its final result while the server
    keeps it, instead of launching it again. Once that result is no longer
    kept, the resent start fails with an error that says the result was lost
    and how the command ended (for example its exit code). It remembers request
    IDs for an hour after their results are dropped, but not across a restart of
    the sandbox tools server. Servers without support are polled as usual.
    """


# ============================================================================
# JSON-RPC Response Types (mirrors server-side types)
# ============================================================================


class _StartResult(BaseModel):
    """Result from exec_remote_start."""

    pid: int


class _PollResult(BaseModel):
    """Result from exec_remote_poll."""

    state: Literal["running", "completed", "killed"]
    exit_code: int | None = None
    seq: int
    stdout: str
    stderr: str


class _WaitResult(BaseModel):
    pid: int
    result: _PollResult


class _ServerReplyError(RuntimeError):
    """The server answered a request with an error, so resending it cannot help."""


class _ServerReplyErrorMapper(GenericJSONRPCErrorMapper):
    """Keep the generic error messages, marking the errors the server returned."""

    @staticmethod
    def server_error(
        code: int, message: str, method: str, params: JSONRPCParamsType
    ) -> Exception:
        del code, method, params
        return _ServerReplyError(message)

    @staticmethod
    def internal_error(
        message: str, method: str, params: JSONRPCParamsType
    ) -> Exception:
        del method, params
        return _ServerReplyError(message)


class _KillResult(BaseModel):
    """Result from exec_remote_kill."""

    seq: int
    stdout: str
    stderr: str


class _WriteStdinResult(BaseModel):
    """Result from exec_remote_write_stdin."""

    seq: int
    stdout: str
    stderr: str


class _CloseStdinResult(BaseModel):
    """Result from exec_remote_close_stdin."""

    seq: int
    stdout: str
    stderr: str


# ============================================================================
# Constants
# ============================================================================

MIN_POLL_INTERVAL = 5
MAX_WAIT_SECONDS = 5.0

WAIT_WINDOW_SECONDS = 60.0
"""Use server-side waits for this long after start, then poll at the interval,
so a long command does not hold a sandbox exec for its whole life."""

POLL_RAMP_SECONDS = (0.25, 0.5, 1.0, 2.0)
"""Waits after the first polls that find a command still running with no output
(not used when the caller sets `poll_interval`). They add polls before the one
the default interval would make, and are skipped when they would not reply
before it. Output does not restart the ramp: once it is used up, polls that
find the command still running with no output are spaced by the interval."""

OUTPUT_REPOLL_SECONDS = 0.5
"""Wait in awaitable mode after a poll that returns output, not a full reply,
when the poll before it did too. Not stretched by `poll_interval`, which spaces
polls that find no output."""

MAX_POLL_OUTPUT_BYTES = 1024 * 1024
"""Most new stdout (and stderr) bytes the sandbox drains for one poll reply (the
sandbox tools' `_MAX_POLL_OUTPUT_BYTES`); a reply that also replays unacknowledged
output can be larger."""

RPC_TIMEOUT = 120
"""Timeout for individual JSON-RPC calls in seconds."""

WAIT_CONCURRENCY_ENV = "INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY"
"""Overrides the limit on held `server_wait` requests in flight."""

SERVER_WAIT_ENV = "INSPECT_EXEC_REMOTE_SERVER_WAIT"
"""Turns `server_wait` on by default when `1` or `true`."""


def _server_wait_default() -> bool:
    return os.environ.get(SERVER_WAIT_ENV, "").strip().lower() in ("1", "true")


START_RESEND_SECONDS = 60.0
"""A failed `server_wait` start is resent only this soon after it was first
sent, well within the 5 minutes the server keeps a completed reply."""

POLL_TIMEOUT_RECOVERY_WAIT_SECONDS: float = 5.0
"""Pause before re-issuing a timed-out poll (see `poll_timeout_recovery`)."""

T = TypeVar("T", bound=BaseModel)


def _wait_concurrency() -> int:
    """Held `server_wait` requests allowed in flight, sized like the Docker CLI limit."""
    return int(os.environ.get(WAIT_CONCURRENCY_ENV, max(effective_cpu_count() * 2, 4)))


class ExecRemoteProcess:
    r"""Handle to a running exec_remote process.

    This class is an async iterator that yields events as they arrive.
    It can only be iterated once (single-use iterator pattern).

    Usage patterns:

    1. Streaming: iterate over the process directly
       ```python
       proc = await sandbox.exec_remote(["cmd"])
       async for event in proc:
           match event:
               case ExecStdout(data=data): print(data)
               case ExecCompleted(exit_code=code): print(f"Done: {code}")
       ```

    2. Fire-and-forget with explicit kill:
       ```python
       proxy = await sandbox.exec_remote(["./proxy"])
       # ... do other work ...
       await proxy.kill()  # terminate when done
       ```

    3. Interactive stdin (requires stdin_open=True):
       ```python
       opts = ExecRemoteStreamingOptions(stdin_open=True)
       proc = await sandbox.exec_remote(["cat"], opts)
       await proc.write_stdin("hello\n")
       await proc.write_stdin("world\n")
       await proc.close_stdin()  # signal EOF
       async for event in proc:
           ...
       ```
    """

    def __init__(
        self,
        sandbox: SandboxEnvironment,
        cmd: list[str],
        options: ExecRemoteStreamingOptions | ExecRemoteCommonOptions,
        sandbox_default_poll_interval: float,
    ) -> None:
        """Initialize an ExecRemoteProcess.

        Args:
            sandbox: The sandbox environment where the process will run.
            cmd: Command and arguments to execute.
            options: Execution options.
            sandbox_default_poll_interval: Default poll interval in seconds,
                provided by the sandbox (e.g. from _default_poll_interval()).
        """
        self._sandbox = sandbox
        self._cmd = cmd
        self._options = options
        self._poll_interval = max(
            MIN_POLL_INTERVAL, options.poll_interval or sandbox_default_poll_interval
        )
        self._use_ramp = options.poll_interval is None
        self._poll_ramp = iter(POLL_RAMP_SECONDS)
        self._awaitable = False
        # awaitable: the last poll returned output, not a full reply
        self._small_output = False
        # when the interval alone would poll next; only a poll that starts then
        # or later stands in for it
        self._interval_due: float | None = None
        # a ramp poll took output the interval's poll would have found
        self._output_before_due = False
        # an extra poll after early output waits for the interval's poll if it
        # would not reply before it
        self._hold_until: float | None = None
        self._pid: int | None = None
        self._last_seq: int = 0
        self._killed = False
        self._completed = False
        self._iteration_started = False
        self._pending_events: list[ExecOutput] = []
        self._request_id: str | None = None
        self._initial_result: _PollResult | None = None
        self._wait_until = 0.0
        self._held = False  # whether the server held the last request
        rpc_timeout = (
            options.poll_timeout if options.poll_timeout is not None else RPC_TIMEOUT
        )
        # A held request keeps at least three quarters of its RPC timeout for
        # the transport (exec startup, queueing, retries of a slow exec).
        self._wait_seconds = min(MAX_WAIT_SECONDS, max(0.0, rpc_timeout / 4))
        self._transport = SandboxJSONRPCTransport(sandbox, SANDBOX_CLI)

    @property
    def pid(self) -> int:
        """Return the process ID."""
        if self._pid is None:
            raise RuntimeError("Process has not been submitted yet")
        return self._pid

    # -------------------------------------------------------------------------
    # RPC helpers
    # -------------------------------------------------------------------------

    async def _rpc(
        self,
        method: str,
        params: dict[str, object],
        result_type: type[T],
        error_mapper: type[JSONRPCErrorMapper] = GenericJSONRPCErrorMapper,
    ) -> T:
        """Make an RPC call to the sandbox."""
        held = bool(params.get("wait_seconds"))
        slot: ConcurrencySemaphore | None = None
        if held and self._options.concurrency:
            slot = await get_or_create_semaphore(
                "exec-remote-wait", _wait_concurrency(), None, False
            )
            if slot.value < 1:
                # Never wait for a free slot: send the request without a hold.
                params, held, slot = {**params, "wait_seconds": 0.0}, False, None
        self._held = held
        extra_args: dict[str, object] = dict(
            timeout=RPC_TIMEOUT
            if self._options.poll_timeout is None
            else self._options.poll_timeout,
            # Run the CLI wrapper as the same user that started the server.
            # When root is available, this is "root" (needed to access the
            # server's private state directory inside the 0700 tools tree).
            # When root isn't available, this is None (sandbox default user).
            user=self._sandbox._tools_user,
            # A held wait idles in the sandbox, so it must not occupy a local
            # subprocess slot that ordinary commands and cleanup need.
            concurrency=self._options.concurrency and not held,
        )
        if self._options.poll_timeout_retry is not None:
            extra_args["timeout_retry"] = self._options.poll_timeout_retry
        async with slot.semaphore if slot is not None else contextlib.nullcontext():
            return await exec_model_request(
                method=method,
                params=params,
                result_type=result_type,
                transport=self._transport,
                error_mapper=error_mapper,
                **extra_args,
            )

    async def _start(self, *, wait: bool = False) -> None:
        """Submit the job to the sandbox."""
        # Build params, converting bytes input to string if needed
        params: dict[str, object] = {"command": shlex.join(self._cmd)}
        if self._options.input is not None:
            if isinstance(self._options.input, bytes):
                params["input"] = self._options.input.decode("utf-8")
            else:
                params["input"] = self._options.input
        if (
            isinstance(self._options, ExecRemoteStreamingOptions)
            and self._options.stdin_open
        ):
            params["stdin_open"] = True
        if self._options.env:
            params["env"] = self._options.env
        if self._options.cwd:
            params["cwd"] = self._options.cwd
        if (user := tools_user_param(self._sandbox, self._options.user)) is not None:
            params["user"] = user

        if (
            wait
            and self._wait_seconds > 0
            and self._sandbox._exec_remote_wait_supported is not False
        ):
            from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy

            self._request_id = uuid4().hex
            # Acknowledge earlier terminal replies so the server can drop them.
            acks, self._sandbox._exec_remote_acks = self._sandbox._exec_remote_acks, ()
            request = {
                **params,
                "request_id": self._request_id,
                "wait_seconds": self._wait_seconds,
            }
            if acks:
                request["acks"] = list(acks)
            sandbox_proxy = cast(SandboxEnvironmentProxy, self._transport.sandbox)
            first_sent = time.monotonic()
            try:
                # Like exec_remote_start, the start is the command's one sandbox
                # event; its reply carries command output, which is logged elsewhere.
                with sandbox_proxy.no_event_output():
                    try:
                        initial = await self._rpc(
                            "exec_remote_start_wait",
                            request,
                            _WaitResult,
                            _ServerReplyErrorMapper,
                        )
                    except Exception as ex:
                        if not self._may_resend_start(ex, first_sent):
                            raise
                        # The server may have launched it before the reply was lost.
                        # A repeated start reports that launch instead of relaunching.
                        initial = await self._rpc(
                            "exec_remote_start_wait",
                            request,
                            _WaitResult,
                            _ServerReplyErrorMapper,
                        )
            except JSONRPCMethodNotFoundError:
                self._sandbox._exec_remote_wait_supported = False
                self._request_id = None
            except _ServerReplyError:
                # The server answered, so it runs nothing that needs a kill.
                self._request_id = None
                raise
            else:
                self._accept_initial_result(initial)
                return

        # Legacy servers cannot be cancelled by request ID; keep the PID reply
        # reachable before acting on an awaitable command's cancellation.
        with anyio.CancelScope(shield=wait):
            result = await self._rpc("exec_remote_start", params, _StartResult)
            self._pid = result.pid
        if wait:
            await anyio.lowlevel.checkpoint_if_cancelled()

    def _may_resend_start(self, ex: Exception, first_sent: float) -> bool:
        """Whether a failed `server_wait` start may be sent again.

        Only a failure that leaves unknown whether the server launched the command
        (a lost connection or reply) is resent, not an error the server returned.
        Timeouts are resent unless `poll_timeout_retry` is False. Nothing is resent
        `START_RESEND_SECONDS` after the first attempt.
        """
        if isinstance(ex, (_ServerReplyError, JSONRPCMethodNotFoundError, ValueError)):
            return False
        if isinstance(ex, TimeoutError) and self._options.poll_timeout_retry is False:
            return False
        return time.monotonic() - first_sent < START_RESEND_SECONDS

    def _accept_initial_result(self, result: _WaitResult) -> None:
        self._sandbox._exec_remote_wait_supported = True
        self._pid = result.pid
        self._initial_result = result.result
        self._wait_until = time.monotonic() + WAIT_WINDOW_SECONDS

    # -------------------------------------------------------------------------
    # Async Iterator Protocol
    # -------------------------------------------------------------------------

    def __aiter__(self) -> "ExecRemoteProcess":
        """Return self as the async iterator.

        This class implements the async iterator protocol directly.
        It can only be iterated once - subsequent iterations will raise RuntimeError.
        """
        if self._iteration_started:
            raise RuntimeError("ExecRemoteProcess can only be iterated once")
        self._iteration_started = True
        return self

    async def __anext__(self) -> ExecOutput:
        """Return the next event from the process.

        Yields ExecStdout and ExecStderr events as output becomes available,
        then yields a final ExecCompleted event when the process terminates.

        Note: After the ExecCompleted event is yielded, the job is automatically
        cleaned up on the server side.

        If cancelled, the process will be killed before re-raising the exception.

        Raises:
            StopAsyncIteration: When the process has completed or been killed.
            RuntimeError: If the process has not been submitted yet.
        """
        if self._pid is None:
            raise RuntimeError("Process has not been submitted yet")

        # Return any pending events first
        if self._pending_events:
            return self._pending_events.pop(0)

        # If already in terminal state, stop iteration
        if self._completed or self._killed:
            raise StopAsyncIteration

        try:
            while True:
                if self._hold_until is not None:
                    hold, self._hold_until = self._hold_until, None
                    await anyio.sleep(max(0.0, hold - anyio.current_time()))
                # Perform the poll
                started = anyio.current_time() if self._use_ramp else 0.0
                result = await self._poll()

                # Collect events from this poll
                events: list[ExecOutput] = []
                if result.stdout:
                    events.append(ExecStdout(data=result.stdout))
                if result.stderr:
                    events.append(ExecStderr(data=result.stderr))

                if result.state != "running" and self._request_id is not None:
                    self._sandbox._exec_remote_acks += (self._request_id,)

                # Check for terminal state
                if result.state == "completed":
                    self._completed = True
                    if result.exit_code is None:
                        raise RuntimeError(
                            "Server returned completed state without exit_code"
                        )
                    events.append(ExecCompleted(exit_code=result.exit_code))
                elif result.state == "killed":
                    # Process was killed (possibly by another call to kill())
                    self._killed = True
                    # Don't yield ExecCompleted for killed processes - kill() discards output

                # If we have events, return the first and queue the rest
                if events:
                    self._pending_events = events[1:]
                    # A held poll already waited in the sandbox: poll again at once.
                    if self._use_ramp and not self._held:
                        if self._interval_due is None or started >= self._interval_due:
                            # the interval's poll: it sees output, polls again at once
                            self._interval_due = anyio.current_time()
                            self._output_before_due = False
                        else:
                            self._output_before_due = True
                            now = anyio.current_time()
                            if now + (now - started) >= self._interval_due:
                                self._hold_until = self._interval_due
                    if self._awaitable and result.state == "running" and not self._held:
                        await self._wait_after_output(result)
                    return events[0]

                # If killed with no events, stop iteration
                if self._killed:
                    raise StopAsyncIteration

                # Still running with no output: unless the server held the
                # request, wait before polling again
                self._small_output = False
                if not self._held:
                    await anyio.sleep(self._idle_wait(started))

        except anyio.get_cancelled_exc_class():
            # Kill the process on cancellation to avoid leaving orphaned processes.
            with anyio.CancelScope(shield=True):
                await self.kill()
            raise

    def _idle_wait(self, started: float) -> float:
        """Seconds to wait after a poll that found the command running with no output.

        The interval alone polls `poll_interval` after such a poll, and at once
        after output. The ramp (POLL_RAMP_SECONDS) adds earlier polls, which do
        not take the place of the interval's: only a poll that starts when the
        interval's would, or later, does. A ramp poll goes out only if, taking
        as long as this one did, it would reply before the interval's poll is
        due; one that still replies after it is followed by another at once.
        If a ramp poll took output the interval's poll would have found, that
        poll is followed by another at once, as it would have been (and the poll
        right after that output waits for the interval's poll if it would not
        reply before it). A `poll_interval` set by the caller (at least 5 s) is
        used without the ramp.
        """
        if not self._use_ramp:
            return self._poll_interval
        now = anyio.current_time()
        due_at = self._interval_due
        if due_at is None or started >= due_at:
            if self._output_before_due:
                # the interval's poll would have found the output a ramp poll
                # took, and polled again at once
                self._output_before_due = False
                return 0.0
            # this poll took the interval's place: the next one is due an interval on
            due_at = self._interval_due = now + self._poll_interval
        elif now >= due_at:
            return 0.0
        due = due_at - now
        step = next(self._poll_ramp, None)
        if step is not None and step + (now - started) < due:
            return step
        return due

    async def _wait_after_output(self, result: _PollResult) -> None:
        """Pace polls after one that found an awaitable command running with output.

        Nobody reads the output before the command ends, so the next poll goes
        out at once only to empty a full reply or to see whether the command
        has ended (the sandbox reports "running" in the poll that returns a
        finished command's last output). While polls in a row return output
        that is not a full reply, wait `OUTPUT_REPOLL_SECONDS` after each, so
        a command that keeps printing is not polled back to back and its end
        is still seen soon after.
        """
        # the sandbox's UTF-8 decoder holds back up to 3 bytes of a split character
        full = any(
            len(output.encode("utf-8")) >= MAX_POLL_OUTPUT_BYTES - 3
            for output in (result.stdout, result.stderr)
        )
        if self._small_output and not full:
            await anyio.sleep(OUTPUT_REPOLL_SECONDS)
        self._small_output = not full

    async def _poll(self) -> _PollResult:
        if self._initial_result is not None:
            result, self._initial_result = self._initial_result, None
            self._last_seq = result.seq
            return result
        recovery_deadline: float | None = None
        retried_repoll_error: RuntimeError | None = None

        def retryable(ex: BaseException) -> bool:
            nonlocal retried_repoll_error
            if not isinstance(ex, RuntimeError):
                return False
            if recovery_deadline is not None:
                # a re-issued poll is not retried once kill() has been called
                if self._killed:
                    return False
                retried_repoll_error = ex
            return True

        @retry(
            wait=wait_exponential_jitter(initial=2),
            stop=(stop_after_attempt(5) | stop_after_delay(30)),
            retry=retry_if_exception(retryable),
            # reraise the underlying RuntimeError on exhaustion so callers and
            # eval logs see the sandbox's own message instead of an opaque
            # RetryError wrapping a Future
            reraise=True,
        )
        async def poll() -> _PollResult:
            from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy

            # tenacity decides to retry before it backs off, so kill() can be
            # called during the backoff: raise the error again instead of polling
            if retried_repoll_error is not None and self._killed:
                raise retried_repoll_error

            sandbox_proxy = cast(SandboxEnvironmentProxy, self._transport.sandbox)
            with sandbox_proxy.no_events():
                if self._request_id is not None:
                    holding = time.monotonic() < self._wait_until
                    reply = await self._rpc(
                        "exec_remote_poll_wait",
                        {
                            "request_id": self._request_id,
                            "ack_seq": self._last_seq,
                            "wait_seconds": self._wait_seconds if holding else 0.0,
                        },
                        _WaitResult,
                    )
                    result = reply.result
                else:
                    result = await self._rpc(
                        "exec_remote_poll",
                        {"pid": self._pid, "ack_seq": self._last_seq},
                        _PollResult,
                    )
            self._last_seq = result.seq
            return result

        # A timed-out poll lost a response, not the process: the server keeps each
        # output chunk until the host acknowledges it (ack_seq), so re-issuing the
        # poll replays what the lost response carried. Each poll() call above is a
        # fresh RuntimeError retry sequence.
        while True:
            try:
                return await poll()
            except TimeoutError as ex:
                recovery = self._options.poll_timeout_recovery
                if recovery is None or self._killed:
                    raise
                now = time.monotonic()
                if recovery_deadline is None:
                    recovery_deadline = now + recovery
                if now >= recovery_deadline:
                    raise
                logger.warning(
                    "exec_remote poll for pid %s timed out (%s); re-polling with "
                    "ack_seq=%s for up to %.0fs more",
                    self._pid,
                    ex,
                    self._last_seq,
                    recovery_deadline - now,
                )
                await anyio.sleep(POLL_TIMEOUT_RECOVERY_WAIT_SECONDS)
                if self._killed:
                    raise
            except RuntimeError as ex:
                if (
                    recovery_deadline is not None
                    and not self._killed
                    and f"No job found with pid {self._pid}" in str(ex)
                ):
                    raise RuntimeError(
                        f"exec_remote process {self._pid} is no longer tracked by the "
                        "sandbox after a poll timed out: it most likely ended during "
                        "the stall, and its final output and exit status were lost "
                        "with the timed-out response (a restart of the sandbox's tool "
                        "server looks the same)."
                    ) from ex
                raise

    def _enqueue_output(self, stdout: str, stderr: str) -> None:
        """Enqueue any non-empty output as pending events for the iterator."""
        if stdout:
            self._pending_events.append(ExecStdout(data=stdout))
        if stderr:
            self._pending_events.append(ExecStderr(data=stderr))

    async def write_stdin(self, data: str | bytes) -> None:
        """Write data to the process's stdin.

        Requires that the process was started with stdin_open=True in
        ExecRemoteStreamingOptions.

        Args:
            data: Data to write. Bytes are decoded to UTF-8.

        Raises:
            RuntimeError: If stdin_open was not set, the process has not
                been started yet, or the process has already terminated.
        """
        if not (
            isinstance(self._options, ExecRemoteStreamingOptions)
            and self._options.stdin_open
        ):
            raise RuntimeError(
                "write_stdin() requires stdin_open=True in ExecRemoteStreamingOptions"
            )
        if self._pid is None:
            raise RuntimeError("Process has not been submitted yet")
        if self._completed or self._killed:
            raise RuntimeError("Cannot write to stdin: process has terminated")

        if isinstance(data, bytes):
            data = data.decode("utf-8")

        result = await self._rpc(
            "exec_remote_write_stdin",
            {"pid": self._pid, "data": data, "ack_seq": self._last_seq},
            _WriteStdinResult,
        )
        self._last_seq = result.seq
        self._enqueue_output(result.stdout, result.stderr)

    async def close_stdin(self) -> None:
        """Close the process's stdin to signal EOF.

        Requires that the process was started with stdin_open=True in
        ExecRemoteStreamingOptions. Idempotent: calling after stdin is already
        closed is a no-op.

        Raises:
            RuntimeError: If stdin_open was not set, or the process has not
                been started yet.
        """
        if not (
            isinstance(self._options, ExecRemoteStreamingOptions)
            and self._options.stdin_open
        ):
            raise RuntimeError(
                "close_stdin() requires stdin_open=True in ExecRemoteStreamingOptions"
            )
        if self._pid is None:
            raise RuntimeError("Process has not been submitted yet")
        if self._completed or self._killed:
            return

        result = await self._rpc(
            "exec_remote_close_stdin",
            {"pid": self._pid, "ack_seq": self._last_seq},
            _CloseStdinResult,
        )
        self._last_seq = result.seq
        self._enqueue_output(result.stdout, result.stderr)

    async def kill(self) -> None:
        """Terminate the process.

        Any output buffered since the last poll is enqueued as pending events
        so the async iterator can yield them before StopAsyncIteration.

        If the process has already completed or been killed, this is a no-op.
        """
        if (
            (self._pid is None and self._request_id is None)
            or self._completed
            or self._killed
        ):
            return

        self._killed = True
        try:
            if self._request_id is not None:
                result = await self._rpc(
                    "exec_remote_kill_wait",
                    {"request_id": self._request_id, "ack_seq": self._last_seq},
                    _KillResult,
                )
                self._sandbox._exec_remote_acks += (self._request_id,)
            else:
                result = await self._rpc(
                    "exec_remote_kill",
                    {"pid": self._pid, "ack_seq": self._last_seq},
                    _KillResult,
                )
            self._last_seq = result.seq
            self._enqueue_output(result.stdout, result.stderr)
        except Exception:
            logging.debug(
                f"exec_remote kill RPC failed for pid {self._pid}", exc_info=True
            )


# ============================================================================
# Factory Functions
# ============================================================================


async def exec_remote_streaming(
    sandbox: SandboxEnvironment,
    cmd: list[str],
    sandbox_default_poll_interval: float,
    options: ExecRemoteStreamingOptions | ExecRemoteCommonOptions | None = None,
) -> ExecRemoteProcess:
    """Create and start an exec_remote process for streaming.

    Submits the start command to the sandbox and returns only after the process
    has been successfully launched (i.e. after the exec_remote_start RPC completes
    and a PID has been assigned).

    Args:
        sandbox: The sandbox environment to run the command in.
        cmd: Command and arguments to execute.
        sandbox_default_poll_interval: Default poll interval in seconds,
            provided by the sandbox (e.g. from _default_poll_interval()).
        options: Execution options.

    Returns:
        ExecRemoteProcess handle that can be iterated for events, or killed.
        The process is guaranteed to have been started when this returns.
    """
    proc = ExecRemoteProcess(
        sandbox,
        cmd,
        options or ExecRemoteCommonOptions(),
        sandbox_default_poll_interval,
    )
    await proc._start()
    return proc


async def exec_remote_awaitable(
    sandbox: SandboxEnvironment,
    cmd: list[str],
    sandbox_default_poll_interval: float,
    options: ExecRemoteAwaitableOptions | ExecRemoteCommonOptions | None = None,
) -> ExecResult[str]:
    """Run a command and return the result without streaming.

    Submits the command, polls until completion, and returns ExecResult.
    With `server_wait`, the server holds requests until output is ready instead.
    If cancelled or timed out, the process will be killed.

    Each output stream (stdout and stderr) is limited to 10 MiB. If output
    exceeds this limit, only the most recent 10 MiB is kept.

    Args:
        sandbox: The sandbox environment to run the command in.
        cmd: Command and arguments to execute.
        sandbox_default_poll_interval: Default poll interval in seconds,
            provided by the sandbox (e.g. from _default_poll_interval()).
        options: Execution options.

    Returns:
        ExecResult[str] with success, returncode, stdout, and stderr.

    Raises:
        TimeoutError: If options.timeout is set and the command exceeds it.
    """
    from .._subprocess import CircularByteBuffer
    from .._subprocess import ExecResult as ExecResultClass
    from .limits import SandboxEnvironmentLimits

    opts = options or ExecRemoteCommonOptions()
    server_wait = isinstance(opts, ExecRemoteAwaitableOptions) and opts.server_wait
    if server_wait:
        proc = ExecRemoteProcess(sandbox, cmd, opts, sandbox_default_poll_interval)
    else:
        proc = await exec_remote_streaming(
            sandbox, cmd, sandbox_default_poll_interval, opts
        )
    proc._awaitable = True

    # Accumulate output chunks with memory limiting
    output_limit = SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE
    stdout_buffer = CircularByteBuffer(output_limit)
    stderr_buffer = CircularByteBuffer(output_limit)

    timeout = opts.timeout if isinstance(opts, ExecRemoteAwaitableOptions) else None

    try:
        with anyio.fail_after(timeout):
            if server_wait:
                await proc._start(wait=True)
            async for event in proc:
                if isinstance(event, ExecStdout):
                    stdout_buffer.write(event.data.encode("utf-8"))
                elif isinstance(event, ExecStderr):
                    stderr_buffer.write(event.data.encode("utf-8"))
                elif isinstance(event, ExecCompleted):
                    return ExecResultClass[str](
                        success=event.success,
                        returncode=event.exit_code,
                        stdout=stdout_buffer.getvalue().decode(
                            "utf-8", errors="replace"
                        ),
                        stderr=stderr_buffer.getvalue().decode(
                            "utf-8", errors="replace"
                        ),
                    )
    except BaseException as ex:
        if proc._request_id is not None or isinstance(
            ex, (TimeoutError, anyio.get_cancelled_exc_class())
        ):
            # Cleanup also covers an ambiguous start whose recovery request failed.
            with anyio.CancelScope(shield=True):
                await proc.kill()
        raise

    # If we get here, the process was killed (no Completed event)
    return ExecResultClass[str](
        success=False,
        returncode=-1,
        stdout=stdout_buffer.getvalue().decode("utf-8", errors="replace"),
        stderr=stderr_buffer.getvalue().decode("utf-8", errors="replace"),
    )
