"""ExecRemote - Asynchronous command execution with streaming output.

This module provides the host-side implementation for exec_remote, enabling
long-running commands in sandbox environments with streaming output.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, Union, cast

import anyio
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from inspect_ai._util._async import current_async_backend
from inspect_ai._util._json_rpc import (
    GenericJSONRPCErrorMapper,
    JSONRPCErrorMapper,
    JSONRPCParamsType,
    exec_model_request,
)
from inspect_ai._util.cpu import effective_cpu_count
from inspect_ai._util.logger import warn_once
from inspect_ai._util.retry import backoff_multiplier
from inspect_ai.util._concurrency import ConcurrencySemaphore, get_or_create_semaphore

from ._cli import tools_user_param
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
    """Interval between poll requests in seconds. The minimum is 5 seconds; 0 or
    `None` uses the sandbox default (also at least 5 seconds). For the first 10
    seconds after the start, sandbox tools that support it hold each poll on the
    server until the command exits or produces output, instead of sleeping
    between polls (not when `stdin_open` is set)."""

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
    function be used to throttle concurrent subprocesses."""

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
    ended during the stall, its exit status is lost and `RuntimeError` is raised.
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
    TimeoutError is raised"""


# ============================================================================
# JSON-RPC Response Types (mirrors server-side types)
# ============================================================================


class _StartResult(BaseModel):
    """Result from exec_remote_start."""

    pid: int
    poll_wait: bool = False
    """The server supports the poll `wait` param. Older servers omit it."""


class _PollResult(BaseModel):
    """Result from exec_remote_poll."""

    state: Literal["running", "completed", "killed"]
    exit_code: int | None = None
    seq: int
    stdout: str
    stderr: str


class _KillResult(BaseModel):
    """Result from exec_remote_kill."""

    seq: int
    stdout: str
    stderr: str


class _KillKeyResult(BaseModel):
    """Result from exec_remote_kill_key."""

    killed: bool


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

RPC_TIMEOUT = 120
"""Timeout for individual JSON-RPC calls in seconds."""

POLL_TIMEOUT_RECOVERY_WAIT_SECONDS: float = 5.0
"""Pause before re-issuing a timed-out poll (see `poll_timeout_recovery`)."""

KILL_KEY_TIMEOUT = 30
"""Bound in seconds on killing a job whose start got no reply."""

SERVER_WAIT_WINDOW_SECONDS: float = 10.0
"""Time after the start reply during which polls wait on the server."""

# METR fork. Everything below is used only when `SERVER_WAIT_ENV` is set, so
# exec_remote polls exactly as upstream does when it is not.

SERVER_WAIT_ENV = "INSPECT_EXEC_REMOTE_SERVER_WAIT"
"""Selects the fork's polling, read when a command starts.

`1` or `true`: polls wait on the server for `EXTENDED_WAIT_WINDOW_SECONDS` after
the start, at most `EXTENDED_MAX_WAIT_SECONDS` each and at most
`WAIT_CONCURRENCY_ENV` of them at once. Any other value: no poll waits on the
server. Either way a poll that did not wait is followed by a pause that grows
with the command's age (`_idle_wait`), an awaitable command that keeps printing
is polled every `OUTPUT_REPOLL_SECONDS`, its kill returns no output, and an
error the server reports for a poll is not retried."""

WAIT_CONCURRENCY_ENV = "INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY"
"""Most polls that wait on the server at once; twice the CPUs, at least 4, if unset."""

EXTENDED_WAIT_WINDOW_SECONDS: float = 60.0
"""Time after the start reply during which polls wait on the server, then they
are spaced by the interval, so a long command does not hold a sandbox exec for
its whole life."""

EXTENDED_MAX_WAIT_SECONDS: float = 5.0
"""Longest a poll waits on the server: a waiting poll holds one of
`WAIT_CONCURRENCY_ENV`'s slots, and short waits pass them round."""

POLL_BACKOFF_SECONDS: float = 0.25
"""Shortest pause after a poll that found the command running with no output."""

OUTPUT_REPOLL_SECONDS: float = 0.5
"""Pause in awaitable mode after a poll that returns output, not a full reply,
when the poll before it did too."""

MAX_POLL_OUTPUT_BYTES = 1024 * 1024
"""Most new stdout (and stderr) bytes the sandbox drains for one poll reply (the
sandbox tools' `_MAX_POLL_OUTPUT_BYTES`)."""


def _server_wait_setting() -> bool | None:
    """`SERVER_WAIT_ENV` as on or off, or `None` when it is unset."""
    value = os.environ.get(SERVER_WAIT_ENV)
    return None if value is None else value.strip().lower() in ("1", "true")


def _wait_concurrency() -> int:
    """Polls allowed to wait on the server at once."""
    # Read the override first: counting CPUs reads cgroup files on every waiting poll.
    value = os.environ.get(WAIT_CONCURRENCY_ENV)
    if value is not None:
        try:
            return max(int(value), 0)
        except ValueError:
            pass
    default = max(effective_cpu_count() * 2, 4)
    if value is not None:
        warn_once(
            logger,
            f"{WAIT_CONCURRENCY_ENV}={value!r} is not an integer; using {default}.",
        )
    return default


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


T = TypeVar("T", bound=BaseModel)


async def _complete_before_cancelling(
    work: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Run `work` to completion, then raise any cancellation of the caller.

    An AnyIO shield does not stop asyncio's `Task.cancel()` (which
    `asyncio.timeout()` and `asyncio.wait_for()` also use), so under asyncio
    `work` runs in its own task and the caller waits for it. `work` shields
    itself from AnyIO cancellation.
    """
    if current_async_backend() != "asyncio":
        await work()
        return

    task = asyncio.ensure_future(work())
    cancelled: asyncio.CancelledError | None = None
    # the shield stops AnyIO from re-delivering its own cancellation
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as ex:
                cancelled = ex
    task.result()
    if cancelled is not None:
        raise cancelled


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
        self._pid: int | None = None
        self._job_key = uuid.uuid4().hex
        self._server_wait_end: float | None = None
        # METR fork (see SERVER_WAIT_ENV); all off until a start finds it set
        self._fork_polling = False
        self._max_wait: float | None = None
        self._started_at = 0.0
        self._held = False  # the last poll waited on the server
        self._awaitable = False  # set by exec_remote_awaitable()
        self._small_output = False  # the last poll returned output, not a full reply
        self._last_seq: int = 0
        self._killed = False
        self._completed = False
        self._iteration_started = False
        self._pending_events: list[ExecOutput] = []
        self._transport = SandboxJSONRPCTransport(sandbox)

    @property
    def pid(self) -> int:
        """Return the process ID."""
        if self._pid is None:
            raise RuntimeError("Process has not been submitted yet")
        return self._pid

    # -------------------------------------------------------------------------
    # RPC helpers
    # -------------------------------------------------------------------------

    @property
    def _rpc_timeout(self) -> float:
        return (
            RPC_TIMEOUT
            if self._options.poll_timeout is None
            else self._options.poll_timeout
        )

    def _server_wait(self) -> float:
        """Seconds the next poll asks the server to wait for exit or output.

        Nonzero only within `SERVER_WAIT_WINDOW_SECONDS` of the start reply of a
        server that supports it. Capped at half the RPC timeout so that the poll
        replies well before it times out.
        """
        if self._server_wait_end is None:
            return 0
        remaining = self._server_wait_end - time.monotonic()
        wait = max(0, min(remaining, self._rpc_timeout / 2))
        return wait if self._max_wait is None else min(wait, self._max_wait)

    async def _rpc(
        self,
        method: str,
        params: dict[str, object],
        result_type: type[T],
        error_mapper: type[JSONRPCErrorMapper] = GenericJSONRPCErrorMapper,
        **overrides: object,
    ) -> T:
        """Make an RPC call to the sandbox.

        `overrides` replace the transport arguments derived from the options
        (timeout, timeout_retry, user, concurrency).
        """
        extra_args: dict[str, object] = dict(
            timeout=self._rpc_timeout,
            # Run the CLI wrapper as the same user that started the server.
            # When root is available, this is "root" (needed to access the
            # server's private state directory inside the 0700 tools tree).
            # When root isn't available, this is None (sandbox default user).
            user=self._sandbox._tools_user,
            concurrency=self._options.concurrency,
        )
        if self._options.poll_timeout_retry is not None:
            extra_args["timeout_retry"] = self._options.poll_timeout_retry
        extra_args.update(overrides)
        return await exec_model_request(
            method=method,
            params=params,
            result_type=result_type,
            transport=self._transport,
            error_mapper=error_mapper,
            **extra_args,
        )

    async def _start(self) -> None:
        """Submit the job to the sandbox.

        The start carries a job key. If the start does not return a PID
        (cancelled, timed out or failed), the job may still have started in
        the sandbox, so it is killed by key before the error propagates.
        """
        # Build params, converting bytes input to string if needed
        params: dict[str, object] = {
            "command": shlex.join(self._cmd),
            "job_key": self._job_key,
        }
        if self._options.input is not None:
            if isinstance(self._options.input, bytes):
                params["input"] = self._options.input.decode("utf-8")
            else:
                params["input"] = self._options.input
        stdin_open = (
            isinstance(self._options, ExecRemoteStreamingOptions)
            and self._options.stdin_open
        )
        if stdin_open:
            params["stdin_open"] = True
        if self._options.env:
            params["env"] = self._options.env
        from .identity import sandbox_identity_cwd

        if cwd := sandbox_identity_cwd(self._sandbox, self._options.cwd):
            params["cwd"] = cwd
        if (user := tools_user_param(self._sandbox, self._options.user)) is not None:
            params["user"] = user

        try:
            result = await self._rpc("exec_remote_start", params, _StartResult)
        except BaseException:
            await self._kill_key()
            raise
        self._pid = result.pid
        self._started_at = time.monotonic()
        setting = _server_wait_setting()
        self._fork_polling = setting is not None
        # A waiting poll that overlaps a stdin write (same ack_seq) can return
        # output twice, so stdin_open processes do not wait on the server.
        if result.poll_wait and not stdin_open and setting is not False:
            window = SERVER_WAIT_WINDOW_SECONDS
            if setting:
                window = EXTENDED_WAIT_WINDOW_SECONDS
                self._max_wait = EXTENDED_MAX_WAIT_SECONDS
            self._server_wait_end = self._started_at + window

    async def _kill_key(self) -> None:
        """Kill the job by key, or make the sandbox refuse its start if it has none yet.

        Runs to completion even if the caller is cancelled, bounded by
        `KILL_KEY_TIMEOUT`, and outside sandbox concurrency limits so it
        cannot wait behind other commands. A failure is logged, as the job
        may then still be running.
        """
        await _complete_before_cancelling(self._kill_key_bounded)

    async def _kill_key_bounded(self) -> None:
        error: BaseException | None = None
        with anyio.move_on_after(KILL_KEY_TIMEOUT, shield=True):
            try:
                await self._rpc(
                    "exec_remote_kill_key",
                    {"job_key": self._job_key},
                    _KillKeyResult,
                    timeout=KILL_KEY_TIMEOUT,
                    timeout_retry=False,
                    concurrency=False,
                )
                return
            except Exception as ex:
                error = ex
        logger.warning(
            "exec_remote could not kill job %s %s (%s); "
            "the command may still be running in the sandbox",
            self._job_key,
            "after its start failed" if self._pid is None else "by its key",
            error or f"timed out after {KILL_KEY_TIMEOUT}s",
        )

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
                # Perform the poll
                result = await self._poll()

                # Collect events from this poll
                events: list[ExecOutput] = []
                if result.stdout:
                    events.append(ExecStdout(data=result.stdout))
                if result.stderr:
                    events.append(ExecStderr(data=result.stderr))

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
                    if self._fork_polling and self._awaitable:
                        if result.state == "running":
                            await self._wait_after_output(result)
                    return events[0]

                # If killed with no events, stop iteration
                if self._killed:
                    raise StopAsyncIteration

                # Still running with no output. Inside the server wait window
                # the next poll waits on the server, so poll again at once,
                # unless this one found no room to wait (see `_poll`).
                self._small_output = False
                if self._server_wait() == 0 or (self._fork_polling and not self._held):
                    await anyio.sleep(self._idle_wait())

        except anyio.get_cancelled_exc_class():
            # Kill the process on cancellation to avoid leaving orphaned processes.
            with anyio.CancelScope(shield=True):
                await self.kill()
            raise

    def _idle_wait(self) -> float:
        """Seconds to pause after a poll that found the command running with no output.

        The interval, as upstream. With the fork's polling, and unless the
        caller set `poll_interval`, as long as the command has run so far,
        from `POLL_BACKOFF_SECONDS` up to the interval: a command that ends
        early is seen within about its own running time, and one that has run
        for a while is polled at the interval.
        """
        if not self._fork_polling or self._options.poll_interval:
            return self._poll_interval
        age = time.monotonic() - self._started_at
        return min(self._poll_interval, max(POLL_BACKOFF_SECONDS, age))

    async def _wait_after_output(self, result: _PollResult) -> None:
        """Pace polls after one that found an awaitable command running with output.

        Nobody reads the output before the command ends, so the next poll goes
        out at once only to empty a full reply or to see whether the command
        has ended. While polls in a row return output that is not a full
        reply, pause `OUTPUT_REPOLL_SECONDS` after each, so a command that
        keeps printing is not polled back to back and its end is still seen
        soon after.
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
        recovery_deadline: float | None = None
        retried_repoll_error: RuntimeError | None = None

        def retryable(ex: BaseException) -> bool:
            nonlocal retried_repoll_error
            if not isinstance(ex, RuntimeError) or isinstance(ex, _ServerReplyError):
                return False
            if recovery_deadline is not None:
                # a re-issued poll is not retried once kill() has been called
                if self._killed:
                    return False
                retried_repoll_error = ex
            return True

        @retry(
            wait=wait_exponential_jitter(**backoff_multiplier(2)),
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

            params: dict[str, object] = {"pid": self._pid, "ack_seq": self._last_seq}
            wait = self._server_wait()
            slot: ConcurrencySemaphore | None = None
            if wait and self._max_wait is not None:
                slot = await get_or_create_semaphore(
                    "exec-remote-wait", _wait_concurrency(), None, False
                )
                if slot.value < 1:
                    warn_once(
                        logger,
                        f"exec_remote polls waiting on the sandbox reached the "
                        f"{WAIT_CONCURRENCY_ENV} limit ({slot.concurrency}): more "
                        "are sent without a wait and paced on the host.",
                    )
                    # Never wait for a free slot: send the poll without a wait.
                    wait, slot = 0, None
            self._held = bool(wait)
            if wait:
                params["wait"] = wait
            overrides: dict[str, object] = {}
            if slot is not None:
                # A waiting poll idles in the sandbox: it must not occupy a local
                # subprocess slot that ordinary commands and cleanup need.
                overrides["concurrency"] = False
            sandbox_proxy = cast(SandboxEnvironmentProxy, self._transport.sandbox)
            async with slot.semaphore if slot is not None else contextlib.nullcontext():
                with sandbox_proxy.no_events():
                    result = await self._rpc(
                        "exec_remote_poll",
                        params,
                        _PollResult,
                        _ServerReplyErrorMapper
                        if self._fork_polling
                        else GenericJSONRPCErrorMapper,
                        **overrides,
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
        if self._pid is None or self._completed or self._killed:
            return

        self._killed = True
        if self._fork_polling and self._awaitable:
            # Nobody takes an awaitable command's output once it is killed, and
            # exec_remote_kill would send all of it: up to 100 MiB a stream.
            await self._kill_key()
            return
        try:
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
    and a PID has been assigned). If the start is cancelled, times out or fails,
    the command is killed in the sandbox before the error propagates.

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
    except TimeoutError:
        # Kill the process (the cancellation handler in __anext__ already
        # does this, but we call it explicitly in case the timeout fires
        # between iterations rather than inside __anext__)
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
