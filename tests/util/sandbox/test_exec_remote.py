"""Tests for exec_remote.

Unit tests mock sandbox.exec() to test host-side logic (guards, kill behavior,
cancellation, accumulation). Integration tests (marked slow) run against a real
Docker container to verify the full host-to-container path.
"""

import ast
import contextlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

import anyio
import anyio.lowlevel
import pytest
import tenacity
from tenacity.wait import wait_none
from test_helpers.utils import skip_if_no_docker

import inspect_ai
import inspect_ai.util._sandbox.exec_remote as exec_remote_module
from inspect_ai import Task, eval
from inspect_ai.analysis import events_df
from inspect_ai.dataset import Sample
from inspect_ai.event import SandboxEvent
from inspect_ai.log import read_eval_log
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool._sandbox_tools_utils.sandbox import (
    SandboxInjectionError,
    _inject_container_tools_code,
)
from inspect_ai.util._sandbox._cli import SANDBOX_CLI
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.environment import SandboxDefaultUser
from inspect_ai.util._sandbox.events import (
    SandboxEnvironmentProxy,
    SandboxTimeoutError,
)
from inspect_ai.util._sandbox.exec_remote import (
    MAX_POLL_OUTPUT_BYTES,
    OUTPUT_REPOLL_SECONDS,
    ExecCompleted,
    ExecRemoteAwaitableOptions,
    ExecRemoteCommonOptions,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecStderr,
    ExecStdout,
    exec_remote_awaitable,
    exec_remote_streaming,
)
from inspect_ai.util._subprocess import ExecResult

# ============================================================================
# Helpers
# ============================================================================


def _rpc(result: dict[str, Any], id: int = 1) -> str:
    """Create a JSON-RPC success response string."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id})


def _start_response(pid: int = 42) -> str:
    """Create a JSON-RPC start response with the given PID."""
    return _rpc({"pid": pid})


def _poll_response(
    state: str = "completed",
    exit_code: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    seq: int = 0,
) -> str:
    """Create a JSON-RPC poll response."""
    return _rpc(
        {
            "state": state,
            "exit_code": exit_code,
            "seq": seq,
            "stdout": stdout,
            "stderr": stderr,
        }
    )


def _kill_response(stdout: str = "", stderr: str = "", seq: int = 0) -> str:
    """Create a JSON-RPC kill response."""
    return _rpc({"seq": seq, "stdout": stdout, "stderr": stderr})


def _write_stdin_response(stdout: str = "", stderr: str = "", seq: int = 0) -> str:
    """Create a JSON-RPC write_stdin response."""
    return _rpc({"seq": seq, "stdout": stdout, "stderr": stderr})


def _close_stdin_response(stdout: str = "", stderr: str = "", seq: int = 0) -> str:
    """Create a JSON-RPC close_stdin response."""
    return _rpc({"seq": seq, "stdout": stdout, "stderr": stderr})


@contextlib.contextmanager
def _no_events_context() -> Iterator[None]:
    """A no-op context manager to stand in for SandboxEnvironmentProxy.no_events()."""
    yield


def _mock_sandbox() -> AsyncMock:
    sandbox = AsyncMock()
    sandbox._tools_default_user = None
    sandbox._exec_remote_wait_supported = False
    sandbox._exec_remote_acks = ()
    sandbox.no_events = _no_events_context
    sandbox.no_event_output = _no_events_context
    return sandbox


def _make_sandbox_mock(responses: list[str]) -> AsyncMock:
    """Create a mock SandboxEnvironment whose exec() returns canned responses.

    Each call to sandbox.exec() pops the next response from the list.
    """
    sandbox = _mock_sandbox()
    sandbox.default_polling_interval.return_value = 5
    sandbox.no_events = _no_events_context

    response_iter = iter(responses)

    async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
        try:
            stdout = next(response_iter)
        except StopIteration:
            raise RuntimeError("Mock sandbox ran out of canned responses")
        return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return sandbox


def _make_never_completing_sandbox() -> AsyncMock:
    """Create a mock sandbox that starts successfully then polls forever as 'running'.

    Useful for testing timeout and cancellation behavior.
    """
    sandbox = _mock_sandbox()
    sandbox.default_polling_interval.return_value = 5
    sandbox.no_events = _no_events_context

    call_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ExecResult(
                success=True, returncode=0, stdout=_start_response(), stderr=""
            )
        else:
            return ExecResult(
                success=True,
                returncode=0,
                stdout=_poll_response(state="running", exit_code=None),
                stderr="",
            )

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return sandbox


def _rpc_error(message: str, code: int = -32099, id: int = 1) -> str:
    """Create a JSON-RPC error response string (default: the server's ToolException code)."""
    return json.dumps(
        {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": id}
    )


def _make_scripted_sandbox(script: list[str | Exception]) -> AsyncMock:
    """Create a mock sandbox whose exec() answers from `script` in order.

    A str entry is returned as stdout; an Exception entry is raised. When the
    script runs out, the last entry repeats. Every JSON-RPC request's method and
    params are recorded in `sandbox.requests`.
    """
    sandbox = _mock_sandbox()
    sandbox.default_polling_interval.return_value = 5
    sandbox.no_events = _no_events_context
    requests: list[tuple[str, dict[str, Any]]] = []
    sandbox.requests = requests
    steps = list(script)

    async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
        await anyio.lowlevel.checkpoint()  # a real request yields to other tasks
        request = json.loads(kwargs["input"])
        requests.append((request["method"], request.get("params", {})))
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Exception):
            raise step
        return ExecResult(success=True, returncode=0, stdout=step, stderr="")

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return sandbox


# ============================================================================
# Single-use iterator
# ============================================================================


class TestSingleUseIterator:
    async def test_second_iteration_raises(self) -> None:
        sandbox = _make_sandbox_mock([_start_response(), _poll_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        _ = [event async for event in proc]

        with pytest.raises(RuntimeError, match="can only be iterated once"):
            async for _ in proc:
                pass


# ============================================================================
# Kill behavior
# ============================================================================


class TestPollRetryExhaustion:
    async def test_poll_retry_exhaustion_reraises_underlying_error(self) -> None:
        sandbox = _mock_sandbox()
        sandbox.default_polling_interval.return_value = 5
        sandbox.no_events = _no_events_context
        sandbox.exec = AsyncMock(
            side_effect=RuntimeError("command terminated with exit code 137")
        )
        proc = ExecRemoteProcess(sandbox, ["cmd"], ExecRemoteCommonOptions(), 5)

        # The retry backoff would take ~30s to exhaust; zero out the wait the
        # same way conftest's fast_retry_waits does for model retries. Patching
        # asyncio.sleep would be a no-op under the trio variant (tenacity routes
        # through its portable sleep helper), and with real sleeps the attempt
        # count assertion becomes timing-sensitive.
        with patch(
            "inspect_ai.util._sandbox.exec_remote.wait_exponential_jitter",
            new=lambda *a, **k: wait_none(),
        ):
            with pytest.raises(RuntimeError, match="exit code 137"):
                await proc._poll()

        # stop_after_attempt(5) pins the attempt count; keep the assertion
        # exact so a stop-config regression is caught
        assert sandbox.exec.call_count == 5


# ============================================================================
# Poll timeout recovery (opt-in re-polling after a poll times out)
# ============================================================================


def _polls(sandbox: AsyncMock) -> list[dict[str, Any]]:
    """The params of every exec_remote_poll request a scripted sandbox received."""
    return [
        params for method, params in sandbox.requests if method == "exec_remote_poll"
    ]


class TestPollTimeoutRecovery:
    @pytest.mark.parametrize(
        "options",
        [ExecRemoteCommonOptions(), ExecRemoteCommonOptions(poll_timeout_retry=False)],
        ids=["default", "poll_timeout_retry_false"],
    )
    async def test_poll_timeout_is_raised_without_recovery(
        self, options: ExecRemoteCommonOptions
    ) -> None:
        """Unless recovery is requested, a timed-out poll raises and is not re-issued."""
        sandbox = _make_scripted_sandbox(
            [
                _start_response(42),
                SandboxTimeoutError("the pod is not answering."),
                _poll_response(state="completed", exit_code=0, seq=0),
            ]
        )
        proc = await exec_remote_streaming(sandbox, ["cmd"], 5, options)

        with pytest.raises(SandboxTimeoutError, match="the pod is not answering"):
            _ = [event async for event in proc]

        assert len(_polls(sandbox)) == 1

    def test_recovery_cannot_override_poll_timeout_retry_false(self) -> None:
        with pytest.raises(ValueError, match="poll_timeout_retry=False"):
            ExecRemoteCommonOptions(poll_timeout_retry=False, poll_timeout_recovery=60)

    async def test_recovery_rides_through_sandbox_timeouts_without_losing_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            exec_remote_module, "POLL_TIMEOUT_RECOVERY_WAIT_SECONDS", 0.0
        )
        stall = SandboxTimeoutError(
            "Command exceeded its 90s timeout and the pod did not report completion "
            "within a further 30s; the pod is not answering."
        )
        sandbox = _make_scripted_sandbox(
            [
                _start_response(42),
                _poll_response(state="running", exit_code=None, stdout="A", seq=1),
                stall,
                stall,
                _poll_response(state="running", exit_code=None, stdout="B", seq=2),
                _poll_response(state="completed", exit_code=0, seq=2),
            ]
        )
        proc = await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=60)
        )

        events = [event async for event in proc]

        assert events == [
            ExecStdout(data="A"),
            ExecStdout(data="B"),
            ExecCompleted(exit_code=0),
        ]
        # The two timed-out polls and the poll that finally answered all asked the
        # server to replay from seq 1: nothing after "A" was acknowledged until "B"
        # arrived.
        assert [params["ack_seq"] for params in _polls(sandbox)] == [0, 1, 1, 1, 2]

    async def test_recovery_time_exhausted_reraises_the_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            exec_remote_module, "POLL_TIMEOUT_RECOVERY_WAIT_SECONDS", 0.05
        )
        sandbox = _make_scripted_sandbox(
            [_start_response(42), SandboxTimeoutError("the pod is not answering.")]
        )
        proc = await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=0.3)
        )

        with pytest.raises(TimeoutError, match="the pod is not answering") as raised:
            await proc._poll()

        assert isinstance(raised.value, SandboxTimeoutError)
        assert 2 <= len(_polls(sandbox)) <= 10

    async def test_cancellation_during_recovery_ends_it_and_kills_the_process(
        self,
    ) -> None:
        """A caller's own deadline cancels recovery at once and still kills the process."""
        sandbox = _make_scripted_sandbox(
            [_start_response(42), SandboxTimeoutError("the pod is not answering.")]
        )

        with pytest.raises(TimeoutError) as raised:
            await exec_remote_awaitable(
                sandbox,
                ["cmd"],
                5,
                ExecRemoteAwaitableOptions(timeout=0.5, poll_timeout_recovery=60),
            )

        # The caller's own deadline, not the sandbox's timeout re-raised once the
        # recovery time ran out.
        assert not isinstance(raised.value, SandboxTimeoutError)
        assert [method for method, _ in sandbox.requests][-1] == "exec_remote_kill"

    async def test_recovery_does_not_reissue_a_timed_out_start(self) -> None:
        """Only polls are re-issued; a sandbox's own timeout retry is outside this mock."""
        sandbox = _make_scripted_sandbox(
            [SandboxTimeoutError("the pod is not answering.")]
        )

        with pytest.raises(TimeoutError):
            await exec_remote_streaming(
                sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=60)
            )

        assert sandbox.exec.call_count == 1

    async def test_poll_after_recovery_names_a_lost_terminal_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            exec_remote_module, "POLL_TIMEOUT_RECOVERY_WAIT_SECONDS", 0.0
        )
        sandbox = _make_scripted_sandbox(
            [
                _start_response(42),
                SandboxTimeoutError("the pod is not answering."),
                _rpc_error("No job found with pid 42"),
            ]
        )
        proc = await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=60)
        )

        # The inner RuntimeError retry would otherwise back off for ~30s.
        with patch(
            "inspect_ai.util._sandbox.exec_remote.wait_exponential_jitter",
            new=lambda *a, **k: wait_none(),
        ):
            with pytest.raises(RuntimeError, match="ended during the stall") as raised:
                await proc._poll()

        assert raised.value.__cause__ is not None
        assert "No job found with pid 42" in str(raised.value.__cause__)

    async def test_no_job_found_without_a_timed_out_poll_stays_the_servers_error(
        self,
    ) -> None:
        """Without a timed-out poll no response was lost, so nothing is relabelled."""
        sandbox = _make_scripted_sandbox(
            [_start_response(42), _rpc_error("No job found with pid 42")]
        )
        proc = await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=60)
        )

        with patch(
            "inspect_ai.util._sandbox.exec_remote.wait_exponential_jitter",
            new=lambda *a, **k: wait_none(),
        ):
            with pytest.raises(RuntimeError, match=r"^No job found with pid 42$"):
                await proc._poll()

    @pytest.mark.parametrize(
        ("kill_during", "kill_reaches_sandbox", "raised", "match"),
        [
            ("pause", True, SandboxTimeoutError, "not answering"),
            ("pause", False, SandboxTimeoutError, "not answering"),
            # the server's own error: the caller killed the process
            ("repoll", True, RuntimeError, r"^No job found with pid 42$"),
            ("repoll", False, SandboxTimeoutError, "not answering"),
            # the re-poll's own error, not the answer to a poll sent after the kill
            ("backoff", True, RuntimeError, "exit code 137"),
            ("backoff", False, RuntimeError, "exit code 137"),
        ],
        ids=[
            "pause-kill_answered",
            "pause-kill_timed_out",
            "repoll-kill_answered",
            "repoll-kill_timed_out",
            "backoff-kill_answered",
            "backoff-kill_timed_out",
        ],
    )
    async def test_no_poll_is_issued_after_kill(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kill_during: str,
        kill_reaches_sandbox: bool,
        raised: type[Exception],
        match: str,
    ) -> None:
        """kill() during recovery stops it, even when the kill request itself timed out.

        The kill comes during the pause before a re-poll, while a re-poll is in
        flight, or during the backoff after a re-poll failed with a `RuntimeError`.
        An in-flight re-poll then gets the server's `No job found` (the kill reached
        the sandbox) or times out (it did not), and is not retried.
        """
        monkeypatch.setattr(
            exec_remote_module, "POLL_TIMEOUT_RECOVERY_WAIT_SECONDS", 0.2
        )
        first_poll_timed_out = anyio.Event()
        repoll_started = anyio.Event()
        repoll_failed = anyio.Event()
        kill_requested = anyio.Event()
        kill_returned = anyio.Event()
        polls = 0
        requests: list[str] = []

        def answer(stdout: str) -> ExecResult[str]:
            return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            nonlocal polls
            method = json.loads(kwargs["input"])["method"]
            requests.append(method)
            if method == "exec_remote_start":
                return answer(_start_response(42))
            if method == "exec_remote_kill":
                kill_requested.set()
                if not kill_reaches_sandbox:
                    raise SandboxTimeoutError("the pod is not answering.")
                return answer(_kill_response())
            polls += 1
            if polls == 1:
                first_poll_timed_out.set()
                raise SandboxTimeoutError("the pod is not answering.")
            if polls == 2 and kill_during == "repoll":
                repoll_started.set()
                await kill_requested.wait()
            if polls == 2 and kill_during == "backoff":
                repoll_failed.set()
                raise RuntimeError("command terminated with exit code 137")
            if kill_requested.is_set() and kill_reaches_sandbox:
                return answer(_rpc_error("No job found with pid 42"))
            raise SandboxTimeoutError("the pod is not answering.")

        sandbox = _mock_sandbox()
        sandbox.default_polling_interval.return_value = 5
        sandbox.no_events = _no_events_context
        sandbox.exec = AsyncMock(side_effect=fake_exec)
        proc = await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_recovery=60)
        )

        kill_after = {
            "pause": first_poll_timed_out,
            "repoll": repoll_started,
            "backoff": repoll_failed,
        }[kill_during]

        async def kill_at_the_chosen_moment() -> None:
            await kill_after.wait()
            await proc.kill()
            kill_returned.set()

        async def backoff_until_killed(seconds: float) -> None:
            await kill_returned.wait()

        def retry_backing_off_until_killed(**kwargs: Any) -> Any:
            return tenacity.retry(sleep=backoff_until_killed, **kwargs)

        # The backoff between RuntimeError retries of a poll lasts until kill() has
        # returned, so a kill always lands inside it and no retry waits ~30s.
        with patch(
            "inspect_ai.util._sandbox.exec_remote.retry",
            new=retry_backing_off_until_killed,
        ):
            with anyio.fail_after(10):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(kill_at_the_chosen_moment)
                    with pytest.raises(raised, match=match):
                        _ = [event async for event in proc]

        assert requests[-1] == "exec_remote_kill"


class TestKill:
    async def test_kill_calls_rpc(self) -> None:
        sandbox = _make_sandbox_mock([_start_response(), _kill_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        await proc.kill()

        assert sandbox.exec.call_count == 2

    async def test_kill_before_start_is_noop(self) -> None:
        sandbox = _mock_sandbox()
        proc = ExecRemoteProcess(sandbox, ["cmd"], ExecRemoteCommonOptions(), 5)

        await proc.kill()
        sandbox.exec.assert_not_called()

    async def test_kill_after_completed_is_noop(self) -> None:
        sandbox = _make_sandbox_mock([_start_response(), _poll_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        _ = [event async for event in proc]
        call_count_before = sandbox.exec.call_count

        await proc.kill()
        assert sandbox.exec.call_count == call_count_before

    async def test_kill_after_kill_is_noop(self) -> None:
        sandbox = _make_sandbox_mock([_start_response(), _kill_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        await proc.kill()
        call_count_before = sandbox.exec.call_count

        await proc.kill()
        assert sandbox.exec.call_count == call_count_before

    async def test_kill_enqueues_remaining_output(self) -> None:
        sandbox = _make_sandbox_mock(
            [_start_response(), _kill_response(stdout="remaining", stderr="errs")]
        )

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        await proc.kill()

        assert proc._pending_events == [
            ExecStdout(data="remaining"),
            ExecStderr(data="errs"),
        ]

    async def test_killed_process_stops_iteration(self) -> None:
        """After external kill, iteration yields remaining output then stops."""
        sandbox = _make_sandbox_mock(
            [
                _start_response(),
                _poll_response(state="killed", exit_code=None, stdout="last"),
            ]
        )

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        events = [event async for event in proc]

        assert events == [ExecStdout(data="last")]

    async def test_kill_suppresses_rpc_exception(self) -> None:
        """kill() should not propagate exceptions from the RPC call.

        Callers (e.g. bridge.py) rely on kill() being safe to call in finally
        blocks without disrupting subsequent cleanup like cancel_scope.cancel().
        """
        sandbox = _mock_sandbox()
        sandbox.default_polling_interval.return_value = 5

        call_count = 0

        async def failing_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Start succeeds
                return ExecResult(
                    success=True, returncode=0, stdout=_start_response(), stderr=""
                )
            # Kill RPC fails (e.g. sandbox transport error)
            raise ConnectionError("sandbox connection lost")

        sandbox.exec = AsyncMock(side_effect=failing_exec)

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

        # Should not raise
        await proc.kill()

        # Should still be marked as killed
        assert proc._killed is True


# ============================================================================
# Cancellation
# ============================================================================


class TestCancellation:
    async def test_cancellation_kills_process(self) -> None:
        """When iteration is cancelled, the process should be killed."""
        kill_called = False
        sandbox = _make_never_completing_sandbox()

        proc = ExecRemoteProcess(sandbox, ["cmd"], ExecRemoteCommonOptions(), 5)
        proc._poll_interval = 0.01
        await proc._start()

        async def mock_kill() -> None:
            nonlocal kill_called
            kill_called = True
            proc._killed = True

        async def iterate() -> None:
            async for _ in proc:
                pass

        with patch.object(proc, "kill", side_effect=mock_kill):
            async with anyio.create_task_group() as tg:
                tg.start_soon(iterate)
                await anyio.sleep(0.05)
                tg.cancel_scope.cancel()

        assert kill_called


# ============================================================================
# Awaitable mode (host-side accumulation logic)
# ============================================================================


class TestAwaitableMode:
    async def test_accumulates_output_across_polls(self) -> None:
        sandbox = _make_sandbox_mock(
            [
                _start_response(),
                _poll_response(state="running", exit_code=None, stdout="a", stderr="x"),
                _poll_response(state="running", exit_code=None, stdout="b", stderr="y"),
                _poll_response(stdout="c", stderr="z"),
            ]
        )

        with patch.object(anyio, "sleep", AsyncMock()):
            result = await exec_remote_awaitable(
                sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_interval=0)
            )

        assert result.success is True
        assert result.stdout == "abc"
        assert result.stderr == "xyz"

    async def test_killed_process_returns_failure(self) -> None:
        """If the process is killed externally, awaitable returns failure."""
        sandbox = _make_sandbox_mock(
            [_start_response(), _poll_response(state="killed", exit_code=None)]
        )

        result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

        assert result.success is False
        assert result.returncode == -1


# ============================================================================
# Poll ramp
# ============================================================================


def _running(stdout: str = "", seq: int = 0) -> str:
    return _poll_response(state="running", exit_code=None, stdout=stdout, seq=seq)


async def _poll_start_times(
    script: Sequence[str | Exception],
    options: ExecRemoteAwaitableOptions | ExecRemoteStreamingOptions | None,
    request_seconds: float | Sequence[float],
    stream: bool = False,
) -> list[float]:
    """Run `script` on a fake clock where requests take `request_seconds`.

    `request_seconds` is one duration for every request (the start included), or
    one per request in order. Returns when each poll (not the start) went out.
    """
    sandbox = _make_scripted_sandbox(list(script))
    scripted_exec = sandbox.exec.side_effect
    times: list[float] = []
    durations = (
        [request_seconds] * len(script)
        if isinstance(request_seconds, float | int)
        else list(request_seconds)
    )

    with _fake_clock() as (_, now):

        async def timed_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            times.append(round(now[0], 6))
            now[0] += durations[len(times) - 1]
            return await scripted_exec(*args, **kwargs)

        sandbox.exec.side_effect = timed_exec
        if stream:
            assert not isinstance(options, ExecRemoteAwaitableOptions)
            proc = await exec_remote_streaming(sandbox, ["cmd"], 5, options)
            _ = [event async for event in proc]
        else:
            assert not isinstance(options, ExecRemoteStreamingOptions)
            await exec_remote_awaitable(sandbox, ["cmd"], 5, options)
    return times[1:]


@contextlib.contextmanager
def _fake_clock() -> Iterator[tuple[AsyncMock, list[float]]]:
    """Patch anyio's sleep and clock so that only sleeps move the clock.

    Yields the sleep mock and a one-item list holding the clock's time, which
    a test can also advance (for example to give each request a duration).
    """
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    with (
        patch.object(anyio, "current_time", lambda: now[0]),
        patch.object(anyio, "sleep", AsyncMock(side_effect=sleep)) as sleep_mock,
    ):
        yield sleep_mock, now


class TestPollRamp:
    """Idle polls start on a short ramp before the default poll interval applies."""

    @pytest.mark.parametrize(
        ("options", "interval"),
        [
            (None, 5),
            (ExecRemoteAwaitableOptions(), 5),
            (ExecRemoteAwaitableOptions(timeout=30), 5),
        ],
        ids=["no_options", "default_options", "other_options"],
    )
    async def test_idle_polls_ramp_up_to_the_default_poll_interval(
        self, options: ExecRemoteAwaitableOptions | None, interval: float
    ) -> None:
        sandbox = _make_scripted_sandbox(
            [_start_response(), *[_running()] * 6, _poll_response(stdout="done")]
        )

        with _fake_clock() as (sleep, _):
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert result.stdout == "done"
        # the ramp's polls come before the one the interval makes at 5 s
        assert [call.args for call in sleep.await_args_list] == [
            (0.25,),
            (0.5,),
            (1.0,),
            (2.0,),
            (interval - 3.75,),
            (interval,),
        ]

    async def test_a_recovered_poll_does_not_use_up_the_ramp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            exec_remote_module, "POLL_TIMEOUT_RECOVERY_WAIT_SECONDS", 0.0
        )
        sandbox = _make_scripted_sandbox(
            [
                _start_response(),
                _running(),
                SandboxTimeoutError("the pod is not answering."),
                _running(),
                _running(),
                _poll_response(stdout="done"),
            ]
        )
        options = ExecRemoteAwaitableOptions(poll_timeout_recovery=60)

        with _fake_clock() as (sleep, _):
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert result.stdout == "done"
        # the re-poll after the timed-out one is part of the same poll, so the
        # ramp carries on where it was
        assert [call.args for call in sleep.await_args_list] == [
            (0.25,),
            (0.0,),
            (0.5,),
            (1.0,),
        ]

    @pytest.mark.parametrize("request_seconds", [0.2, 0.35, 0.4, 1.2, 2.5])
    async def test_ramp_adds_polls_without_delaying_the_interval_polls(
        self, request_seconds: float
    ) -> None:
        """With requests that take time, every poll the interval alone makes still happens."""
        script = [_start_response(), *[_running()] * 12, _poll_response(stdout="done")]
        interval_only = await _poll_start_times(
            script, ExecRemoteAwaitableOptions(poll_interval=5), request_seconds
        )
        with_ramp = await _poll_start_times(
            script, ExecRemoteAwaitableOptions(), request_seconds
        )

        assert set(interval_only[:3]) <= set(with_ramp)
        if request_seconds == 0.2:
            # the ramp's polls, then the interval's: 5 s after the reply before each
            assert with_ramp[:7] == [0.2, 0.65, 1.35, 2.55, 4.75, 5.4, 10.6]
            assert interval_only[:3] == [0.2, 5.4, 10.6]

    async def test_a_ramp_poll_that_replies_after_the_interval_poll_is_followed_at_once(
        self,
    ) -> None:
        """A ramp poll slower than the one before it cannot stand in for the interval's."""
        script = [_start_response(), *[_running()] * 12, _poll_response(stdout="done")]
        # 0.2 s requests, except the fifth poll's, which takes 1 s
        durations = [0.2] * 5 + [1.0] + [0.2] * 10

        with_ramp = await _poll_start_times(
            script, ExecRemoteAwaitableOptions(), durations
        )

        # the poll at 4.75 replies at 5.75, after the interval's 5.4: poll again then
        assert with_ramp[:7] == [0.2, 0.65, 1.35, 2.55, 4.75, 5.75, 10.95]

    @pytest.mark.parametrize("stream", [True, False], ids=["streaming", "awaitable"])
    async def test_poll_after_early_output_does_not_cross_the_interval_poll(
        self, stream: bool
    ) -> None:
        """With steady requests, the poll after a ramp poll's output keeps the interval's time."""
        script = [
            _start_response(),
            *[_running()] * 3,
            _running("a", seq=1),
            _running(seq=1),
            _running("b", seq=2),
            *[_running(seq=2)] * 4,
            _poll_response(stdout="done", seq=3),
        ]
        options = None if stream else ExecRemoteAwaitableOptions()

        times = await _poll_start_times(script, options, 1.0, stream=stream)

        # "a" comes from the poll at 5.75, and a poll at once would reply after
        # the interval's poll at 7.0: wait for that, then poll as the interval
        # alone would
        assert times[:7] == [1.0, 2.25, 3.75, 5.75, 7.0, 8.0, 9.0]

    @pytest.mark.parametrize("stream", [True, False], ids=["streaming", "awaitable"])
    async def test_output_at_the_interval_poll_clears_the_early_output(
        self, stream: bool
    ) -> None:
        """The interval's poll finding output itself repolls once, not twice."""
        script = [
            _start_response(),
            _running(),
            _running("a", seq=1),
            *[_running(seq=1)] * 4,
            _running("b", seq=2),
            *[_running(seq=2)] * 4,
            _poll_response(stdout="done", seq=3),
        ]
        options = None if stream else ExecRemoteAwaitableOptions()

        times = await _poll_start_times(script, options, 0.2, stream=stream)

        assert times[:9] == [0.2, 0.65, 0.85, 1.55, 2.75, 4.95, 5.4, 5.6, 10.8]

    @pytest.mark.parametrize("stream", [True, False], ids=["streaming", "awaitable"])
    async def test_output_from_a_ramp_poll_keeps_the_interval_poll(
        self, stream: bool
    ) -> None:
        """Output seen early keeps the interval's poll, which then polls again at once.

        The interval alone would have found the output at its poll and polled
        again at once, so a command ending just after it is not seen later.
        """
        script = [
            _start_response(),
            _running(),
            _running("a", seq=1),
            *[_running(seq=1)] * 8,
            _poll_response(stdout="done", seq=2),
        ]
        options = None if stream else ExecRemoteAwaitableOptions()

        times = await _poll_start_times(script, options, 0.2, stream=stream)

        # "a" came from the poll at 0.65; the interval's poll at 5.4 still
        # happens, followed at once by the one it would make after output
        assert times[:9] == [0.2, 0.65, 0.85, 1.55, 2.75, 4.95, 5.4, 5.6, 10.8]

    @pytest.mark.parametrize(
        ("poll_interval", "interval"),
        [(1, 5), (5, 5), (12, 12)],
        ids=["below_minimum", "minimum", "above_minimum"],
    )
    async def test_explicit_poll_interval_skips_the_ramp(
        self, poll_interval: float, interval: float
    ) -> None:
        sandbox = _make_scripted_sandbox(
            [_start_response(), *[_running()] * 3, _poll_response(stdout="done")]
        )
        options = ExecRemoteAwaitableOptions(poll_interval=poll_interval)

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert result.stdout == "done"
        assert [call.args for call in sleep.await_args_list] == [(interval,)] * 3

    async def test_streaming_with_explicit_poll_interval_skips_the_ramp(self) -> None:
        sandbox = _make_scripted_sandbox(
            [_start_response(), _running(), _poll_response(stdout="done")]
        )
        options = ExecRemoteStreamingOptions(poll_interval=10)

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            proc = await exec_remote_streaming(sandbox, ["cmd"], 5, options)
            events = [event async for event in proc]

        assert [type(event) for event in events] == [ExecStdout, ExecCompleted]
        assert [call.args for call in sleep.await_args_list] == [(10,)]

    @pytest.mark.parametrize(
        ("stream", "waits"),
        [
            # one wait per idle poll, continuing the ramp across output polls,
            # the last up to the interval's poll at 5 s (output seen by the
            # ramp's polls does not move it)
            (True, [0.25, 0.5, 1.0, 2.0, 1.25]),
            # and a short one, outside the ramp, after "c" (output again)
            (False, [0.25, 0.5, OUTPUT_REPOLL_SECONDS, 1.0, 2.0, 0.75]),
        ],
        ids=["streaming", "awaitable"],
    )
    async def test_output_polls_repoll_at_once_without_using_the_ramp(
        self, stream: bool, waits: list[float]
    ) -> None:
        sandbox = _make_scripted_sandbox(
            [
                _start_response(),
                _running(),
                _running("a", seq=1),
                _running(seq=1),
                _running("b", seq=2),
                _running("c", seq=3),
                _running(seq=3),
                _running(seq=3),
                _running("d", seq=4),
                _running(seq=4),
                _poll_response(stdout="e", seq=5),
            ]
        )

        with _fake_clock() as (sleep, _):
            if stream:
                proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
                events = [event async for event in proc]
                stdout = "".join(e.data for e in events if isinstance(e, ExecStdout))
            else:
                stdout = (await exec_remote_awaitable(sandbox, ["cmd"], 5)).stdout

        assert stdout == "abcde"
        assert len(_polls(sandbox)) == 10
        assert [call.args for call in sleep.await_args_list] == [(w,) for w in waits]


# ============================================================================
# Awaitable output polls
# ============================================================================


def _chatty_script(count: int) -> list[str | Exception]:
    """A command that has printed a line by every poll, then ends."""
    return [
        _start_response(),
        *[_running(f"{i}\n", seq=i) for i in range(1, count)],
        _poll_response(stdout=f"{count}\n", seq=count),
    ]


class TestAwaitableOutputPolls:
    """Awaitable mode re-polls at once after output only to see the end or empty a full reply."""

    @pytest.mark.parametrize(
        "options",
        [None, ExecRemoteAwaitableOptions(poll_interval=12)],
        ids=["default", "poll_interval"],
    )
    async def test_chatty_command_waits_after_small_output(
        self, options: ExecRemoteAwaitableOptions | None
    ) -> None:
        sandbox = _make_scripted_sandbox(_chatty_script(6))

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert result.stdout == "".join(f"{i}\n" for i in range(1, 7))
        assert len(_polls(sandbox)) == 6
        # at once after the first output poll, then a short wait after each one
        # (not the ramp or poll_interval, which space polls with no output)
        assert [call.args for call in sleep.await_args_list] == [
            (OUTPUT_REPOLL_SECONDS,)
        ] * 4

    async def test_streaming_chatty_command_repolls_at_once(self) -> None:
        sandbox = _make_scripted_sandbox(_chatty_script(6))

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
            events = [event async for event in proc]

        assert len(events) == 7
        assert len(_polls(sandbox)) == 6
        sleep.assert_not_awaited()

    @pytest.mark.parametrize(
        "options",
        [None, ExecRemoteAwaitableOptions(poll_interval=12)],
        ids=["default", "poll_interval"],
    )
    async def test_end_is_seen_at_once_after_output(
        self, options: ExecRemoteAwaitableOptions | None
    ) -> None:
        # the sandbox reports "running" in the poll that returns the last output
        sandbox = _make_scripted_sandbox(
            [_start_response(), _running("out", seq=1), _poll_response(seq=1)]
        )

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert result.stdout == "out"
        assert len(_polls(sandbox)) == 2
        sleep.assert_not_awaited()

    async def test_end_after_output_in_polls_in_a_row_is_seen_after_one_wait(
        self,
    ) -> None:
        # the second "running" reply carries the last output, so the poll that
        # sees the end comes OUTPUT_REPOLL_SECONDS later
        sandbox = _make_scripted_sandbox(
            [
                _start_response(),
                _running("a", seq=1),
                _running("b", seq=2),
                _poll_response(seq=2),
            ]
        )

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

        assert result.stdout == "ab"
        assert len(_polls(sandbox)) == 3
        assert [call.args for call in sleep.await_args_list] == [
            (OUTPUT_REPOLL_SECONDS,)
        ]

    def test_full_reply_size_matches_the_sandbox_tools(self) -> None:
        """MAX_POLL_OUTPUT_BYTES mirrors the server's cap, so a full reply counts as full."""
        job_source = (
            Path(inspect_ai.__file__).parents[1]
            / "inspect_sandbox_tools"
            / "src"
            / "inspect_sandbox_tools"
            / "_remote_tools"
            / "_exec_remote"
            / "_job.py"
        )
        if not job_source.is_file():
            pytest.skip(f"no sibling inspect_sandbox_tools source at {job_source}")

        def value(node: ast.expr) -> int:
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult), (
                ast.dump(node)
            )
            return value(node.left) * value(node.right)

        caps = [
            value(node.value)
            for node in ast.parse(job_source.read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_MAX_POLL_OUTPUT_BYTES"
                for target in node.targets
            )
        ]
        assert caps == [MAX_POLL_OUTPUT_BYTES]

    @pytest.mark.parametrize(
        ("stdout", "stderr", "waits"),
        [
            ("x" * MAX_POLL_OUTPUT_BYTES, "", []),
            ("", "x" * MAX_POLL_OUTPUT_BYTES, []),
            # a character split at the cap is held back until the next reply
            ("x" * (MAX_POLL_OUTPUT_BYTES - 3), "", []),
            ("\u00e9" * (MAX_POLL_OUTPUT_BYTES // 2), "", []),
            ("x" * (MAX_POLL_OUTPUT_BYTES - 4), "", [OUTPUT_REPOLL_SECONDS] * 3),
        ],
        ids=["stdout", "stderr", "held_back", "multibyte", "not_full"],
    )
    async def test_full_reply_repolls_at_once(
        self, stdout: str, stderr: str, waits: list[float]
    ) -> None:
        def reply(seq: int) -> str:
            return _poll_response(
                state="running", exit_code=None, stdout=stdout, stderr=stderr, seq=seq
            )

        sandbox = _make_scripted_sandbox(
            [
                _start_response(),
                _running("a", seq=1),
                reply(2),
                reply(3),
                _running("b", seq=4),
                _poll_response(seq=4),
            ]
        )

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

        assert result.stdout == "a" + stdout * 2 + "b"
        assert len(_polls(sandbox)) == 5
        assert [call.args for call in sleep.await_args_list] == [(w,) for w in waits]

    @pytest.mark.parametrize("stop", ["timeout", "cancel"])
    async def test_timeout_or_cancellation_in_the_wait_kills_the_command(
        self, stop: str
    ) -> None:
        sandbox = _make_scripted_sandbox([_start_response(), _running("1\n", seq=1)])
        in_wait = anyio.Event()

        async def wait_until_cancelled(seconds: float) -> None:
            assert seconds == OUTPUT_REPOLL_SECONDS
            in_wait.set()
            await anyio.Event().wait()

        with patch.object(anyio, "sleep", wait_until_cancelled), anyio.fail_after(10):
            if stop == "timeout":
                with pytest.raises(TimeoutError):
                    await exec_remote_awaitable(
                        sandbox, ["cmd"], 5, ExecRemoteAwaitableOptions(timeout=0.5)
                    )
            else:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(exec_remote_awaitable, sandbox, ["cmd"], 5)
                    await in_wait.wait()
                    tg.cancel_scope.cancel()

        assert in_wait.is_set()
        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start",
            "exec_remote_poll",
            "exec_remote_poll",
            "exec_remote_kill",
        ]


# ============================================================================
# Server-side waits (server_wait)
# ============================================================================


def _wait_response(
    state: str = "completed",
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    seq: int = 1,
) -> str:
    return _rpc(
        {
            "pid": 42,
            "result": {
                "state": state,
                "exit_code": exit_code,
                "seq": seq,
                "stdout": stdout,
                "stderr": stderr,
            },
        }
    )


_WAIT = ExecRemoteAwaitableOptions(server_wait=True)


class TestAwaitableWaitProtocol:
    async def test_fast_command_returns_output_and_exit_in_one_request(self) -> None:
        sandbox = _make_scripted_sandbox(
            [_wait_response(stdout="contents\n", stderr="warning\n", exit_code=7)]
        )
        sandbox._exec_remote_wait_supported = None

        result = await exec_remote_awaitable(sandbox, ["cat", "file"], 5, _WAIT)

        assert (result.stdout, result.stderr, result.returncode, result.success) == (
            "contents\n",
            "warning\n",
            7,
            False,
        )
        assert len(sandbox.requests) == 1
        method, params = sandbox.requests[0]
        assert method == "exec_remote_start_wait"
        assert params["command"] == "cat file"
        assert params["wait_seconds"] == 5.0
        assert sandbox.exec.call_args.kwargs["concurrency"] is False

    async def test_start_is_the_one_sandbox_event_and_omits_output(self) -> None:
        """Like exec_remote_start, one event shows the command line; output is not logged twice."""
        sandbox = _make_scripted_sandbox(
            [
                _wait_response(state="running", exit_code=None, stdout="a", seq=1),
                _wait_response(stdout="b", seq=2),
            ]
        )
        proxy = SandboxEnvironmentProxy(sandbox)

        with patch("inspect_ai.log._transcript.transcript") as transcript:
            result = await exec_remote_awaitable(proxy, ["cmd", "x y"], 5, _WAIT)

        assert result.stdout == "ab"
        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_poll_wait",
        ]
        [event] = [
            call.args[0] for call in transcript.return_value._event.call_args_list
        ]
        assert (event.action, event.cmd, event.result, event.output) == (
            "exec",
            f"{SANDBOX_CLI} exec",
            0,
            None,
        )
        request = json.loads(event.input)
        assert request["method"] == "exec_remote_start_wait"
        assert request["params"]["command"] == "cmd 'x y'"

    async def test_waiting_polls_do_not_add_a_client_sleep(self) -> None:
        sandbox = _make_scripted_sandbox(
            [
                _wait_response(state="running", exit_code=None, stdout="first", seq=1),
                _wait_response(state="running", exit_code=None, seq=2),
                _wait_response(stdout="last", seq=3),
            ]
        )
        sandbox._exec_remote_wait_supported = None

        with patch.object(anyio, "sleep", side_effect=AssertionError("extra sleep")):
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert result.stdout == "firstlast"
        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_poll_wait",
            "exec_remote_poll_wait",
        ]
        assert [params["ack_seq"] for _, params in sandbox.requests[1:]] == [1, 2]
        assert all(params["wait_seconds"] == 5.0 for _, params in sandbox.requests[1:])
        assert not any(
            call.kwargs["concurrency"] for call in sandbox.exec.call_args_list
        )

    async def test_hold_leaves_most_of_the_rpc_timeout_to_the_transport(self) -> None:
        sandbox = _make_scripted_sandbox([_wait_response()])
        sandbox._exec_remote_wait_supported = None
        options = ExecRemoteAwaitableOptions(poll_timeout=8, server_wait=True)

        await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert sandbox.requests[0][1]["wait_seconds"] == 2.0
        assert sandbox.exec.call_args.kwargs["timeout"] == 8

    async def test_held_replies_with_output_add_no_ramp_or_pacing_sleep(self) -> None:
        """The sandbox already waited: poll again at once, even with the ramp on."""
        sandbox = _make_scripted_sandbox(
            [
                _wait_response(state="running", exit_code=None, stdout="a", seq=1),
                _wait_response(state="running", exit_code=None, stdout="b", seq=2),
                _wait_response(stdout="c", seq=3),
            ]
        )
        sandbox._exec_remote_wait_supported = None

        with patch.object(anyio, "sleep", side_effect=AssertionError("extra sleep")):
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert result.stdout == "abc"

    async def test_server_wait_env_reaches_a_call_without_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(exec_remote_module.SERVER_WAIT_ENV, "1")
        sandbox = _make_scripted_sandbox([_wait_response(stdout="out")])
        sandbox._exec_remote_wait_supported = None

        result = await exec_remote_awaitable(sandbox, ["cmd"], 5, None)

        assert result.stdout == "out"
        assert [method for method, _ in sandbox.requests] == ["exec_remote_start_wait"]

    async def test_server_wait_env_sets_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(exec_remote_module.SERVER_WAIT_ENV, "true")
        assert ExecRemoteAwaitableOptions().server_wait
        assert not ExecRemoteAwaitableOptions(server_wait=False).server_wait
        monkeypatch.setenv(exec_remote_module.SERVER_WAIT_ENV, "0")
        assert not ExecRemoteAwaitableOptions().server_wait
        monkeypatch.delenv(exec_remote_module.SERVER_WAIT_ENV)
        assert not ExecRemoteAwaitableOptions().server_wait

    async def test_no_free_held_slot_sends_requests_at_once_without_a_hold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other commands' holds never delay a start or poll; it is sent unheld."""
        monkeypatch.setenv("INSPECT_EXEC_REMOTE_WAIT_CONCURRENCY", "1")
        sandbox = _mock_sandbox()
        sandbox._exec_remote_wait_supported = None
        held, release = anyio.Event(), anyio.Event()
        commands: dict[str, str] = {}
        replies = [
            _wait_response(state="running", exit_code=None, seq=1),
            _wait_response(stdout="quick", seq=2),
        ]
        sent: list[tuple[str, str, float, bool]] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            request = json.loads(kwargs["input"])
            params = request["params"]
            command = commands.setdefault(params["request_id"], params.get("command"))
            sent.append(
                (
                    request["method"],
                    command,
                    params["wait_seconds"],
                    kwargs["concurrency"],
                )
            )
            if command == "slow":
                held.set()
                await release.wait()
                response = _wait_response(stdout="slow")
            else:
                response = replies.pop(0)
            return ExecResult(success=True, returncode=0, stdout=response, stderr="")

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            async with anyio.create_task_group() as group:
                group.start_soon(exec_remote_awaitable, sandbox, ["slow"], 5, _WAIT)
                await held.wait()
                # completes while the slow command still holds the only slot
                with anyio.fail_after(5):
                    quick = await exec_remote_awaitable(sandbox, ["quick"], 5, _WAIT)
                release.set()

        assert quick.stdout == "quick"
        assert sent == [
            ("exec_remote_start_wait", "slow", 5.0, False),
            ("exec_remote_start_wait", "quick", 0.0, True),
            ("exec_remote_poll_wait", "quick", 0.0, True),
        ]
        # an unheld reply of a running command is followed by the poll ramp
        assert [call.args for call in sleep.await_args_list] == [
            (exec_remote_module.POLL_RAMP_SECONDS[0],)
        ]

    async def test_waits_stop_after_the_wait_window(self) -> None:
        sandbox = _make_scripted_sandbox(
            [
                _wait_response(state="running", exit_code=None, seq=1),
                _wait_response(state="running", exit_code=None, seq=2),
                _wait_response(stdout="done", seq=3),
            ]
        )
        sandbox._exec_remote_wait_supported = None

        with (
            patch.object(exec_remote_module, "WAIT_WINDOW_SECONDS", 0.0),
            patch.object(anyio, "sleep", AsyncMock()) as sleep,
        ):
            result = await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert result.stdout == "done"
        # the held start is followed by a poll at once; unheld polls by the ramp
        assert [call.args for call in sleep.await_args_list] == [
            (exec_remote_module.POLL_RAMP_SECONDS[0],)
        ]
        assert [params["wait_seconds"] for _, params in sandbox.requests] == [
            5.0,
            0.0,
            0.0,
        ]
        assert [call.kwargs["concurrency"] for call in sandbox.exec.call_args_list] == [
            False,
            True,
            True,
        ]

    @pytest.mark.parametrize(
        "error",
        [
            SandboxTimeoutError("lost reply"),
            ConnectionError("connection reset"),
            RuntimeError("Sandbox.exec failure executing exec_remote_start_wait"),
        ],
    )
    async def test_lost_start_reply_is_recovered_by_repeating_the_start(
        self, error: Exception
    ) -> None:
        sandbox = _make_scripted_sandbox([error, _wait_response(stdout="once")])
        sandbox._exec_remote_wait_supported = None

        result = await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert result.stdout == "once"
        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_start_wait",
        ]
        assert sandbox.requests[0][1] == sandbox.requests[1][1]

    async def test_terminal_replies_are_acknowledged_on_the_next_start(self) -> None:
        """With nothing to acknowledge a start has no `acks`."""
        sandbox = _make_scripted_sandbox([_wait_response(stdout="first")])
        sandbox._exec_remote_wait_supported = None

        await exec_remote_awaitable(sandbox, ["one"], 5, _WAIT)
        await exec_remote_awaitable(sandbox, ["two"], 5, _WAIT)

        first, second = (params for _, params in sandbox.requests)
        assert "acks" not in first
        assert second["acks"] == [first["request_id"]]
        assert sandbox._exec_remote_acks == (second["request_id"],)

    async def test_kill_replies_are_acknowledged_on_the_next_start(self) -> None:
        sandbox = _mock_sandbox()
        sandbox._exec_remote_wait_supported = None
        requests: list[dict[str, Any]] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            request = json.loads(kwargs["input"])
            requests.append(request)
            if len(requests) == 1:
                await anyio.sleep_forever()
            if request["method"] == "exec_remote_kill_wait":
                response = _kill_response(seq=1)
            else:
                response = _wait_response()
            return ExecResult(success=True, returncode=0, stdout=response, stderr="")

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        with pytest.raises(TimeoutError):
            await exec_remote_awaitable(
                sandbox,
                ["one"],
                5,
                ExecRemoteAwaitableOptions(timeout=0.01, server_wait=True),
            )
        await exec_remote_awaitable(sandbox, ["two"], 5, _WAIT)

        first, kill, second = (request["params"] for request in requests)
        assert kill["request_id"] == first["request_id"]
        assert second["acks"] == [first["request_id"]]

    async def test_old_server_fallback_is_cached(self) -> None:
        sandbox = _make_scripted_sandbox(
            [
                _rpc_error("Method not found", code=-32601),
                _start_response(),
                _poll_response(stdout="old"),
                _start_response(),
                _poll_response(stdout="old again"),
            ]
        )
        sandbox._exec_remote_wait_supported = None

        assert (await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)).stdout == "old"
        assert (
            await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)
        ).stdout == "old again"
        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_start",
            "exec_remote_poll",
            "exec_remote_start",
            "exec_remote_poll",
        ]

    async def test_command_error_cannot_trigger_legacy_start(self) -> None:
        sandbox = _make_scripted_sandbox([_rpc_error("Method not found", code=-32099)])
        sandbox._exec_remote_wait_supported = None

        with pytest.raises(RuntimeError, match="Method not found"):
            await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert [method for method, _ in sandbox.requests] == ["exec_remote_start_wait"]

    async def test_launch_error_is_raised_once_with_the_legacy_text(self) -> None:
        """The server answered, so the start is neither resent nor killed."""
        legacy = "FileNotFoundError(2, 'No such file or directory')"
        sandbox = _make_scripted_sandbox(
            [_rpc_error(legacy, code=-32098), _kill_response()]
        )
        sandbox._exec_remote_wait_supported = None

        with pytest.raises(RuntimeError) as raised:
            await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert str(raised.value) == legacy
        assert [method for method, _ in sandbox.requests] == ["exec_remote_start_wait"]

    async def test_timed_out_start_is_not_resent_without_timeout_retries(
        self,
    ) -> None:
        sandbox = _make_scripted_sandbox(
            [SandboxTimeoutError("lost reply"), _kill_response()]
        )
        sandbox._exec_remote_wait_supported = None
        options = ExecRemoteAwaitableOptions(poll_timeout_retry=False, server_wait=True)

        with pytest.raises(TimeoutError):
            await exec_remote_awaitable(sandbox, ["cmd"], 5, options)

        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_kill_wait",
        ]

    async def test_start_is_not_resent_after_the_resend_window(self) -> None:
        sandbox = _make_scripted_sandbox(
            [ConnectionError("connection reset"), _kill_response()]
        )
        sandbox._exec_remote_wait_supported = None

        with (
            patch.object(exec_remote_module, "START_RESEND_SECONDS", 0.0),
            pytest.raises(ConnectionError),
        ):
            await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_kill_wait",
        ]

    async def test_failed_start_recovery_still_kills_the_owned_request(self) -> None:
        sandbox = _make_scripted_sandbox(
            [
                SandboxTimeoutError("lost start"),
                RuntimeError("recovery failed"),
                _kill_response(),
            ]
        )
        sandbox._exec_remote_wait_supported = None

        with pytest.raises(RuntimeError, match="recovery failed"):
            await exec_remote_awaitable(sandbox, ["cmd"], 5, _WAIT)

        assert [method for method, _ in sandbox.requests] == [
            "exec_remote_start_wait",
            "exec_remote_start_wait",
            "exec_remote_kill_wait",
        ]
        assert len({params["request_id"] for _, params in sandbox.requests}) == 1

    async def test_legacy_start_keeps_pid_available_for_cancellation(self) -> None:
        sandbox = _mock_sandbox()
        started, release = anyio.Event(), anyio.Event()
        requests: list[dict[str, Any]] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            request = json.loads(kwargs["input"])
            requests.append(request)
            if request["method"] == "exec_remote_start":
                started.set()
                await release.wait()
                response = _start_response()
            else:
                assert request["method"] == "exec_remote_kill"
                response = _kill_response()
            return ExecResult(success=True, returncode=0, stdout=response, stderr="")

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        async with anyio.create_task_group() as group:
            group.start_soon(exec_remote_awaitable, sandbox, ["cmd"], 5, _WAIT)
            await started.wait()
            group.cancel_scope.cancel()
            release.set()

        assert [request["method"] for request in requests] == [
            "exec_remote_start",
            "exec_remote_kill",
        ]
        assert requests[-1]["params"]["pid"] == 42

    @pytest.mark.parametrize("options", [None, ExecRemoteAwaitableOptions(timeout=30)])
    async def test_server_waits_are_off_by_default(
        self, options: ExecRemoteAwaitableOptions | None
    ) -> None:
        """A supporting server gets exactly the polling requests sent without waits."""
        sandbox = _make_scripted_sandbox(
            [
                _start_response(),
                _poll_response(state="running", exit_code=None),
                _poll_response(stdout="out", seq=1),
            ]
        )
        sandbox._exec_remote_wait_supported = None
        sandbox._tools_user = "root"

        with patch.object(anyio, "sleep", AsyncMock()) as sleep:
            result = await exec_remote_awaitable(sandbox, ["cmd", "a b"], 5, options)

        assert result.stdout == "out"
        # without server waits, the poll ramp checks again after its first step
        assert [call.args for call in sleep.await_args_list] == [
            (exec_remote_module.POLL_RAMP_SECONDS[0],)
        ]
        exec_kwargs = {
            "env": {"INSPECT_SANDBOX_JSON_RPC_RESPONSE_MAX_BYTES": "10485760"},
            "timeout": 120,
            "timeout_retry": True,
            "user": "root",
            "concurrency": True,
        }
        requests = [
            ("exec_remote_start", {"command": "cmd 'a b'"}),
            ("exec_remote_poll", {"pid": 42, "ack_seq": 0}),
            ("exec_remote_poll", {"pid": 42, "ack_seq": 0}),
        ]
        sent = [
            (call.args, {**call.kwargs, "input": json.loads(call.kwargs["input"])})
            for call in sandbox.exec.call_args_list
        ]
        for _, kwargs in sent:
            del kwargs["input"]["id"]
        assert sent == [
            (
                ([SANDBOX_CLI, "exec"],),
                {
                    **exec_kwargs,
                    "input": {"jsonrpc": "2.0", "method": method, "params": params},
                },
            )
            for method, params in requests
        ]

    async def test_cancellation_during_start_kills_by_request_id(self) -> None:
        sandbox = _mock_sandbox()
        sandbox._exec_remote_wait_supported = None
        started = anyio.Event()
        requests: list[dict[str, Any]] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            request = json.loads(kwargs["input"])
            requests.append(request)
            if request["method"] == "exec_remote_start_wait":
                started.set()
                await anyio.sleep_forever()
            assert request["method"] == "exec_remote_kill_wait"
            return ExecResult(
                success=True, returncode=0, stdout=_kill_response(), stderr=""
            )

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        async with anyio.create_task_group() as group:
            group.start_soon(exec_remote_awaitable, sandbox, ["cmd"], 5, _WAIT)
            await started.wait()
            group.cancel_scope.cancel()

        assert [request["method"] for request in requests] == [
            "exec_remote_start_wait",
            "exec_remote_kill_wait",
        ]
        assert (
            requests[0]["params"]["request_id"] == requests[1]["params"]["request_id"]
        )

    async def test_deadline_includes_initial_wait(self) -> None:
        sandbox = _mock_sandbox()
        sandbox._exec_remote_wait_supported = None
        requests: list[str] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            method = json.loads(kwargs["input"])["method"]
            requests.append(method)
            if method == "exec_remote_start_wait":
                await anyio.sleep_forever()
            assert method == "exec_remote_kill_wait"
            return ExecResult(
                success=True, returncode=0, stdout=_kill_response(), stderr=""
            )

        sandbox.exec = AsyncMock(side_effect=fake_exec)
        with pytest.raises(TimeoutError):
            await exec_remote_awaitable(
                sandbox,
                ["cmd"],
                5,
                ExecRemoteAwaitableOptions(timeout=0.01, server_wait=True),
            )
        assert requests == ["exec_remote_start_wait", "exec_remote_kill_wait"]


@pytest.mark.parametrize("log_format", ["eval", "json"])
def test_start_event_without_output_reads_back_from_a_log(
    tmp_path: Path, log_format: Literal["eval", "json"]
) -> None:
    """Log readers and the event dataframe, unchanged here, accept `output=None`."""

    @solver
    def server_wait_command() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            sandbox = _make_scripted_sandbox([_wait_response(stdout="out")])
            sandbox._exec_remote_wait_supported = None
            proxy = SandboxEnvironmentProxy(sandbox)
            result = await exec_remote_awaitable(proxy, ["cat", "f"], 5, _WAIT)
            assert result.stdout == "out"
            return state

        return solve

    task = Task(dataset=[Sample(input="x")], solver=server_wait_command())
    [log] = eval(
        task,
        model="mockllm/model",
        log_dir=str(tmp_path),
        log_format=log_format,
        display="none",
    )

    assert log.status == "success" and log.location
    [sample] = read_eval_log(log.location).samples or []
    [event] = [event for event in sample.events if isinstance(event, SandboxEvent)]
    assert (event.action, event.cmd, event.result, event.output) == (
        "exec",
        f"{SANDBOX_CLI} exec",
        0,
        None,
    )
    assert json.loads(event.input or "")["method"] == "exec_remote_start_wait"
    events = events_df(log.location, filter=lambda e: e.event == "sandbox", quiet=True)
    assert events["event"].tolist() == ["sandbox"]


# ============================================================================
# Timeout
# ============================================================================


class TestTimeout:
    async def test_timeout_raises_timeout_error(self) -> None:
        """Awaitable mode raises TimeoutError when timeout expires."""
        sandbox = _make_never_completing_sandbox()

        with pytest.raises(TimeoutError):
            await exec_remote_awaitable(
                sandbox,
                ["sleep", "999"],
                5,
                ExecRemoteAwaitableOptions(timeout=1, poll_interval=0.1),
            )

    async def test_timeout_kills_process(self) -> None:
        """On timeout, the process should be killed."""
        sandbox = _mock_sandbox()
        sandbox.default_polling_interval.return_value = 5
        sandbox.no_events = _no_events_context

        call_count = 0
        methods_called: list[str] = []

        async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
            nonlocal call_count
            call_count += 1

            input_str = kwargs.get("input", "")
            if input_str:
                payload = json.loads(input_str)
                methods_called.append(payload.get("method", ""))

            if call_count == 1:
                return ExecResult(
                    success=True, returncode=0, stdout=_start_response(), stderr=""
                )
            elif methods_called and methods_called[-1] == "exec_remote_kill":
                return ExecResult(
                    success=True, returncode=0, stdout=_kill_response(), stderr=""
                )
            else:
                return ExecResult(
                    success=True,
                    returncode=0,
                    stdout=_poll_response(state="running", exit_code=None),
                    stderr="",
                )

        sandbox.exec = AsyncMock(side_effect=fake_exec)

        with pytest.raises(TimeoutError):
            await exec_remote_awaitable(
                sandbox,
                ["sleep", "999"],
                5,
                ExecRemoteAwaitableOptions(timeout=1, poll_interval=0.1),
            )

        assert "exec_remote_kill" in methods_called


# ============================================================================
# PID access
# ============================================================================


class TestPidAccess:
    def test_pid_before_start_raises(self) -> None:
        proc = ExecRemoteProcess(AsyncMock(), ["cmd"], ExecRemoteCommonOptions(), 5)
        with pytest.raises(RuntimeError, match="not been submitted"):
            _ = proc.pid

    async def test_pid_after_start(self) -> None:
        sandbox = _make_sandbox_mock([_start_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        assert proc.pid == 42

    async def test_anext_before_start_raises(self) -> None:
        proc = ExecRemoteProcess(AsyncMock(), ["cmd"], ExecRemoteCommonOptions(), 5)
        proc._iteration_started = True
        with pytest.raises(RuntimeError, match="not been submitted"):
            await proc.__anext__()


class TestUserParam:
    _DEFAULT_USER = SandboxDefaultUser(uid=1111, gid=1111, groups=[1111], home="/h")

    async def _start_params(
        self, options: ExecRemoteStreamingOptions
    ) -> dict[str, Any]:
        sandbox = _make_sandbox_mock([_start_response()])
        sandbox._tools_default_user = self._DEFAULT_USER
        await exec_remote_streaming(sandbox, ["cmd"], 5, options)
        params: dict[str, Any] = json.loads(sandbox.exec.call_args.kwargs["input"])[
            "params"
        ]
        return params

    async def test_default_user_identity_sent_without_explicit_user(self) -> None:
        params = await self._start_params(ExecRemoteStreamingOptions())
        assert params["user"] == self._DEFAULT_USER._asdict()

    async def test_explicit_user_wins(self) -> None:
        params = await self._start_params(ExecRemoteStreamingOptions(user="nobody"))
        assert params["user"] == "nobody"


# ============================================================================
# write_stdin / close_stdin error guards
# ============================================================================


class TestWriteStdin:
    async def test_write_stdin_without_stdin_open_raises(self) -> None:
        """write_stdin raises RuntimeError when stdin_open is False."""
        sandbox = _make_sandbox_mock([_start_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        with pytest.raises(
            RuntimeError, match="stdin_open=True in ExecRemoteStreamingOptions"
        ):
            await proc.write_stdin("data")

    async def test_write_stdin_before_start_raises(self) -> None:
        """write_stdin raises RuntimeError when process not started."""
        proc = ExecRemoteProcess(
            AsyncMock(), ["cmd"], ExecRemoteStreamingOptions(stdin_open=True), 5
        )
        with pytest.raises(RuntimeError, match="not been submitted"):
            await proc.write_stdin("data")

    async def test_write_stdin_after_completed_raises(self) -> None:
        """write_stdin raises RuntimeError after process has completed."""
        sandbox = _make_sandbox_mock([_start_response(), _poll_response()])
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5, opts)
        _ = [event async for event in proc]

        with pytest.raises(RuntimeError, match="process has terminated"):
            await proc.write_stdin("data")

    async def test_write_stdin_after_killed_raises(self) -> None:
        """write_stdin raises RuntimeError after process has been killed."""
        sandbox = _make_sandbox_mock([_start_response(), _kill_response()])
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5, opts)
        await proc.kill()

        with pytest.raises(RuntimeError, match="process has terminated"):
            await proc.write_stdin("data")

    async def test_write_stdin_enqueues_output(self) -> None:
        """Output returned from write_stdin is enqueued as pending events."""
        sandbox = _make_sandbox_mock(
            [
                _start_response(),
                _write_stdin_response(stdout="chunk1", stderr="err1"),
                _poll_response(),
            ]
        )
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cat"], 5, opts)
        await proc.write_stdin("hello")

        events = [event async for event in proc]
        assert any(isinstance(e, ExecStdout) and e.data == "chunk1" for e in events)
        assert any(isinstance(e, ExecStderr) and e.data == "err1" for e in events)


# ============================================================================
# Poll timeout and retry option plumbing
# ============================================================================


class TestPollTimeoutOptions:
    """Verify that poll_timeout and poll_timeout_retry propagate correctly."""

    async def test_default_poll_timeout_uses_rpc_timeout(self) -> None:
        """Default poll_timeout=None falls back to RPC_TIMEOUT (30)."""
        sandbox = _make_sandbox_mock([_start_response()])

        await exec_remote_streaming(sandbox, ["cmd"], 5, ExecRemoteCommonOptions())

        kwargs = sandbox.exec.call_args_list[0].kwargs
        assert kwargs["timeout"] == 120

    async def test_explicit_poll_timeout_propagates(self) -> None:
        """Explicit poll_timeout value is passed through to sandbox.exec."""
        sandbox = _make_sandbox_mock([_start_response()])

        await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout=60)
        )

        kwargs = sandbox.exec.call_args_list[0].kwargs
        assert kwargs["timeout"] == 60

    async def test_default_poll_timeout_retry_preserves_transport_default(self) -> None:
        """Default poll_timeout_retry=None should not pass timeout_retry.

        Transport uses its own default (True).
        """
        sandbox = _make_sandbox_mock([_start_response()])

        await exec_remote_streaming(sandbox, ["cmd"], 5, ExecRemoteCommonOptions())

        kwargs = sandbox.exec.call_args_list[0].kwargs
        # timeout_retry should either not be present (transport defaults to True)
        # or be True
        assert kwargs.get("timeout_retry", True) is True

    async def test_explicit_poll_timeout_retry_false_propagates(self) -> None:
        """Explicit poll_timeout_retry=False disables retries."""
        sandbox = _make_sandbox_mock([_start_response()])

        await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_retry=False)
        )

        kwargs = sandbox.exec.call_args_list[0].kwargs
        assert kwargs["timeout_retry"] is False

    async def test_explicit_poll_timeout_retry_true_propagates(self) -> None:
        """Explicit poll_timeout_retry=True is passed through."""
        sandbox = _make_sandbox_mock([_start_response()])

        await exec_remote_streaming(
            sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_timeout_retry=True)
        )

        kwargs = sandbox.exec.call_args_list[0].kwargs
        assert kwargs["timeout_retry"] is True


# ============================================================================
# Options positional order
# ============================================================================

_BASE_POSITIONAL: tuple[object, ...] = (
    "in",
    "/work",
    {"K": "V"},
    "someone",
    7.0,
    30.0,
    False,
    False,
)


def _base_fields(options: ExecRemoteCommonOptions) -> tuple[object, ...]:
    return (
        options.input,
        options.cwd,
        options.env,
        options.user,
        options.poll_interval,
        options.poll_timeout,
        options.poll_timeout_retry,
        options.concurrency,
    )


class TestOptionsPositionalOrder:
    """New options are keyword-only, so positional callers keep their meaning."""

    def test_common_options(self) -> None:
        options = ExecRemoteCommonOptions(
            "in", "/work", {"K": "V"}, "someone", 7.0, 30.0, False, False
        )

        assert _base_fields(options) == _BASE_POSITIONAL
        assert options.poll_timeout_recovery is None

    def test_streaming_options(self) -> None:
        options = ExecRemoteStreamingOptions(
            "in", "/work", {"K": "V"}, "someone", 7.0, 30.0, False, False, True
        )

        assert _base_fields(options) == _BASE_POSITIONAL
        assert options.stdin_open is True
        assert options.poll_timeout_recovery is None

    def test_awaitable_options(self) -> None:
        options = ExecRemoteAwaitableOptions(
            "in", "/work", {"K": "V"}, "someone", 7.0, 30.0, False, False, 12.0
        )

        assert _base_fields(options) == _BASE_POSITIONAL
        assert options.timeout == 12.0
        assert options.poll_timeout_recovery is None


# ============================================================================
# close_stdin error guards
# ============================================================================


class TestCloseStdin:
    async def test_close_stdin_without_stdin_open_raises(self) -> None:
        """close_stdin raises RuntimeError when stdin_open is False."""
        sandbox = _make_sandbox_mock([_start_response()])

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
        with pytest.raises(
            RuntimeError, match="stdin_open=True in ExecRemoteStreamingOptions"
        ):
            await proc.close_stdin()

    async def test_close_stdin_before_start_raises(self) -> None:
        """close_stdin raises RuntimeError when process not started."""
        proc = ExecRemoteProcess(
            AsyncMock(), ["cmd"], ExecRemoteStreamingOptions(stdin_open=True), 5
        )
        with pytest.raises(RuntimeError, match="not been submitted"):
            await proc.close_stdin()

    async def test_close_stdin_after_completed_is_noop(self) -> None:
        """close_stdin is a no-op after process has completed."""
        sandbox = _make_sandbox_mock([_start_response(), _poll_response()])
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5, opts)
        _ = [event async for event in proc]
        call_count_before = sandbox.exec.call_count

        await proc.close_stdin()
        assert sandbox.exec.call_count == call_count_before

    async def test_close_stdin_enqueues_output(self) -> None:
        """Output returned from close_stdin is enqueued as pending events."""
        sandbox = _make_sandbox_mock(
            [
                _start_response(),
                _close_stdin_response(stdout="final_out", stderr="final_err"),
                _poll_response(),
            ]
        )
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cat"], 5, opts)
        await proc.close_stdin()

        events = [event async for event in proc]
        assert any(isinstance(e, ExecStdout) and e.data == "final_out" for e in events)
        assert any(isinstance(e, ExecStderr) and e.data == "final_err" for e in events)

    async def test_close_stdin_after_killed_is_noop(self) -> None:
        """close_stdin is a no-op after process has been killed."""
        sandbox = _make_sandbox_mock([_start_response(), _kill_response()])
        opts = ExecRemoteStreamingOptions(stdin_open=True)

        proc = await exec_remote_streaming(sandbox, ["cmd"], 5, opts)
        await proc.kill()
        call_count_before = sandbox.exec.call_count

        await proc.close_stdin()
        assert sandbox.exec.call_count == call_count_before


# ============================================================================
# Integration tests (real Docker container)
# ============================================================================


@pytest.fixture
async def docker_sandbox(request):
    """Yield a proxy-wrapped Docker sandbox with tools injected."""
    task_name = f"{__name__}_{request.node.name}"

    await DockerSandboxEnvironment.task_init(task_name=task_name, config=None)
    envs = await DockerSandboxEnvironment.sample_init(
        task_name=task_name, config=None, metadata={}
    )

    async def cleanup() -> None:
        await DockerSandboxEnvironment.sample_cleanup(
            task_name=task_name, config=None, environments=envs, interrupted=False
        )
        await DockerSandboxEnvironment.task_cleanup(
            task_name=task_name, config=None, cleanup=True
        )

    raw = envs["default"]
    try:
        await _inject_container_tools_code(raw)
    except (FileNotFoundError, SandboxInjectionError):
        await cleanup()
        pytest.skip("Sandbox tools binary not available")

    proxy = SandboxEnvironmentProxy(raw)
    proxy._tools_injected = True

    # Smoke test: verify the injected binary accepts the current RPC schema.
    # Fails when the binary predates host-side schema changes (e.g. ack_seq).
    # Only skip on ValueError (response-validation/schema mismatch). RuntimeError
    # — including poll-retry exhaustion, which is now reraised — must propagate
    # so a genuinely broken docker exec fails loudly instead of silently skipping
    # this whole integration suite.
    try:
        await exec_remote_awaitable(proxy, ["true"], proxy.default_polling_interval())
    except ValueError:
        await cleanup()
        pytest.skip("Injected binary incompatible with current host-side RPC schema")

    try:
        yield proxy
    finally:
        await cleanup()


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_echo(docker_sandbox) -> None:
    """Streaming exec_remote yields stdout then completed."""
    proc = await exec_remote_streaming(
        docker_sandbox, ["echo", "hello"], docker_sandbox.default_polling_interval()
    )
    assert isinstance(proc, ExecRemoteProcess)

    events = [event async for event in proc]

    stdout_events = [e for e in events if isinstance(e, ExecStdout)]
    completed = [e for e in events if isinstance(e, ExecCompleted)]
    assert len(completed) == 1
    assert completed[0].exit_code == 0
    assert "hello" in "".join(e.data for e in stdout_events)


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_stderr(docker_sandbox) -> None:
    """Streaming exec_remote captures stderr."""
    proc = await exec_remote_streaming(
        docker_sandbox,
        ["sh", "-c", "echo err >&2"],
        docker_sandbox.default_polling_interval(),
    )

    events = [event async for event in proc]

    stderr_events = [e for e in events if isinstance(e, ExecStderr)]
    assert "err" in "".join(e.data for e in stderr_events)


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_nonzero_exit(docker_sandbox) -> None:
    """Non-zero exit code is reported in ExecCompleted."""
    proc = await exec_remote_streaming(
        docker_sandbox,
        ["sh", "-c", "exit 42"],
        docker_sandbox.default_polling_interval(),
    )

    events = [event async for event in proc]

    completed = [e for e in events if isinstance(e, ExecCompleted)]
    assert len(completed) == 1
    assert completed[0].exit_code == 42


@skip_if_no_docker
@pytest.mark.slow
async def test_awaitable_echo(docker_sandbox) -> None:
    """Awaitable exec_remote returns ExecResult."""
    result = await exec_remote_awaitable(
        docker_sandbox, ["echo", "hello"], docker_sandbox.default_polling_interval()
    )

    assert result.success
    assert "hello" in result.stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_awaitable_failure(docker_sandbox) -> None:
    """Awaitable exec_remote reports failure."""
    result = await exec_remote_awaitable(
        docker_sandbox,
        ["sh", "-c", "echo oops >&2; exit 1"],
        docker_sandbox.default_polling_interval(),
    )

    assert not result.success
    assert result.returncode == 1
    assert "oops" in result.stderr


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_multiline(docker_sandbox) -> None:
    """Streaming collects multi-line output correctly."""
    proc = await exec_remote_streaming(
        docker_sandbox,
        ["sh", "-c", "echo line1; echo line2; echo line3"],
        docker_sandbox.default_polling_interval(),
    )

    events = [event async for event in proc]

    stdout = "".join(e.data for e in events if isinstance(e, ExecStdout))
    assert "line1" in stdout
    assert "line2" in stdout
    assert "line3" in stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_kill(docker_sandbox) -> None:
    """Kill terminates a long-running process."""
    proc = await exec_remote_streaming(
        docker_sandbox, ["sleep", "300"], docker_sandbox.default_polling_interval()
    )
    await proc.kill()
    assert proc._killed


@skip_if_no_docker
@pytest.mark.slow
async def test_awaitable_timeout(docker_sandbox) -> None:
    """Awaitable mode raises TimeoutError when timeout expires."""
    with pytest.raises(TimeoutError):
        await exec_remote_awaitable(
            docker_sandbox,
            ["sleep", "300"],
            docker_sandbox.default_polling_interval(),
            ExecRemoteAwaitableOptions(timeout=2, poll_interval=0.5),
        )


@skip_if_no_docker
@pytest.mark.slow
async def test_streaming_stdin(docker_sandbox) -> None:
    """write_stdin sends data to the process and close_stdin triggers EOF."""
    opts = ExecRemoteStreamingOptions(stdin_open=True)
    proc = await exec_remote_streaming(
        docker_sandbox, ["cat"], docker_sandbox.default_polling_interval(), opts
    )

    await proc.write_stdin("hello from stdin\n")
    await proc.close_stdin()

    events = [event async for event in proc]

    stdout = "".join(e.data for e in events if isinstance(e, ExecStdout))
    assert "hello from stdin" in stdout

    completed = [e for e in events if isinstance(e, ExecCompleted)]
    assert len(completed) == 1
    assert completed[0].exit_code == 0


@skip_if_no_docker
@pytest.mark.slow
async def test_env_vars(docker_sandbox) -> None:
    """Environment variables are passed to the process."""
    opts = ExecRemoteCommonOptions(env={"MY_TEST_VAR": "hello123"})
    result = await exec_remote_awaitable(
        docker_sandbox,
        ["sh", "-c", "echo $MY_TEST_VAR"],
        docker_sandbox.default_polling_interval(),
        opts,
    )

    assert result.success
    assert "hello123" in result.stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_cwd(docker_sandbox) -> None:
    """Working directory is respected."""
    opts = ExecRemoteCommonOptions(cwd="/tmp")
    result = await exec_remote_awaitable(
        docker_sandbox, ["pwd"], docker_sandbox.default_polling_interval(), opts
    )

    assert result.success
    assert "/tmp" in result.stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_input_string(docker_sandbox) -> None:
    """String input is passed to the process's stdin."""
    opts = ExecRemoteCommonOptions(input="hello from input\n")
    result = await exec_remote_awaitable(
        docker_sandbox, ["cat"], docker_sandbox.default_polling_interval(), opts
    )

    assert result.success
    assert "hello from input" in result.stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_input_bytes(docker_sandbox) -> None:
    """Bytes input is decoded to UTF-8 and passed to stdin."""
    opts = ExecRemoteCommonOptions(input=b"bytes input\n")
    result = await exec_remote_awaitable(
        docker_sandbox, ["cat"], docker_sandbox.default_polling_interval(), opts
    )

    assert result.success
    assert "bytes input" in result.stdout


@skip_if_no_docker
@pytest.mark.slow
async def test_ack_seq_retransmit(docker_sandbox) -> None:
    """Resetting _last_seq simulates a lost response; server retransmits."""
    import anyio

    opts = ExecRemoteStreamingOptions(stdin_open=True)
    proc = await exec_remote_streaming(
        docker_sandbox, ["cat"], docker_sandbox.default_polling_interval(), opts
    )

    # Write and poll until cat echoes back, so the server has output to track
    await proc.write_stdin("hello\n")
    collected: list[ExecStdout | ExecStderr | ExecCompleted] = list(
        proc._pending_events
    )
    proc._pending_events.clear()
    for _ in range(20):
        result = await proc._poll()
        if result.stdout:
            collected.append(ExecStdout(data=result.stdout))
            break
        await anyio.sleep(0.1)

    assert any(isinstance(e, ExecStdout) and "hello" in e.data for e in collected), (
        f"Expected 'hello' in output, got {collected}"
    )
    seq_after_first = proc._last_seq
    assert seq_after_first > 0

    # Simulate lost response: reset ack_seq so server thinks we missed everything
    proc._last_seq = 0

    # Next write — server retransmits "hello" (unacked) plus new "world"
    await proc.write_stdin("world\n")
    retransmit_events = list(proc._pending_events)
    proc._pending_events.clear()
    for _ in range(20):
        result = await proc._poll()
        if result.stdout:
            retransmit_events.append(ExecStdout(data=result.stdout))
            break
        await anyio.sleep(0.1)

    retransmit_stdout = "".join(
        e.data for e in retransmit_events if isinstance(e, ExecStdout)
    )
    assert "hello" in retransmit_stdout, (
        f"Expected retransmit of unacked 'hello', got: {retransmit_stdout!r}"
    )
    assert "world" in retransmit_stdout

    await proc.close_stdin()
    _ = [event async for event in proc]


def _record_methods(fail_first: bool = False) -> tuple[list[str], Any]:
    """Wrap exec_model_request to record methods, optionally losing the first reply."""
    from inspect_ai._util._json_rpc import exec_model_request as original

    methods: list[str] = []

    async def record(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        methods.append(kwargs["method"])
        if fail_first and len(methods) == 1:
            raise ConnectionError("connection dropped after the server replied")
        return result

    return methods, patch.object(exec_remote_module, "exec_model_request", record)


@skip_if_no_docker
@pytest.mark.slow
async def test_awaitable_returns_in_one_wait_trip(docker_sandbox) -> None:
    """Output around a pause and a child's late output return with the exit status."""
    methods, recording = _record_methods()
    with recording:
        result = await exec_remote_awaitable(
            docker_sandbox,
            ["sh", "-c", "echo early; sleep 1; (sleep 0.3; echo late) & echo err >&2"],
            docker_sandbox.default_polling_interval(),
            _WAIT,
        )

    assert (result.stdout, result.stderr) == ("early\nlate\n", "err\n")
    assert methods == ["exec_remote_start_wait"]


@skip_if_no_docker
@pytest.mark.slow
async def test_lost_start_reply_does_not_launch_twice(docker_sandbox) -> None:
    methods, recording = _record_methods(fail_first=True)
    with recording:
        result = await exec_remote_awaitable(
            docker_sandbox,
            ["sh", "-c", "echo x >> /tmp/launches; cat /tmp/launches"],
            docker_sandbox.default_polling_interval(),
            _WAIT,
        )
    launches = await docker_sandbox.exec(["cat", "/tmp/launches"])

    assert methods == ["exec_remote_start_wait", "exec_remote_start_wait"]
    assert result.stdout == launches.stdout == "x\n"


@skip_if_no_docker
@pytest.mark.slow
async def test_awaitable_polls_without_server_wait(docker_sandbox) -> None:
    methods, recording = _record_methods()
    with recording:
        result = await exec_remote_awaitable(
            docker_sandbox, ["echo", "hi"], docker_sandbox.default_polling_interval()
        )

    assert result.stdout == "hi\n"
    assert methods[0] == "exec_remote_start"
    assert set(methods[1:]) == {"exec_remote_poll"}
