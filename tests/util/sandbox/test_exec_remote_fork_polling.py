"""METR fork: exec_remote's polling when INSPECT_EXEC_REMOTE_SERVER_WAIT is set.

Unset, exec_remote polls as upstream does, which test_exec_remote.py covers.
"""

import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from tenacity import wait_none
from test_exec_remote import (
    _RUNNING,
    _make_scripted_sandbox,
    _mock_sandbox,
    _no_events_context,
    _poll_response,
    _poll_wait_start_response,
    _polls,
    _rpc,
    _rpc_error,
    _start_response,
)

from inspect_ai.util._concurrency import get_or_create_semaphore, init_concurrency
from inspect_ai.util._sandbox import exec_remote as er
from inspect_ai.util._sandbox.exec_remote import (
    ExecCompleted,
    ExecRemoteAwaitableOptions,
    ExecRemoteCommonOptions,
    ExecRemoteProcess,
    ExecStdout,
    exec_remote_awaitable,
    exec_remote_streaming,
)
from inspect_ai.util._subprocess import ExecResult


@pytest.fixture(autouse=True)
def fresh_semaphores() -> None:
    init_concurrency()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the duration of each `anyio.sleep` call and return at once."""
    recorded: list[float] = []
    real_sleep = anyio.sleep

    async def record_sleep(delay: float) -> None:
        recorded.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(anyio, "sleep", record_sleep)
    return recorded


@pytest.fixture
def waits_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(er.SERVER_WAIT_ENV, "1")
    monkeypatch.delenv(er.WAIT_CONCURRENCY_ENV, raising=False)


@pytest.fixture
def waits_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(er.SERVER_WAIT_ENV, "0")


def _output(text: str) -> str:
    return _poll_response(state="running", exit_code=None, stdout=text)


# --- the wait window -----------------------------------------------------------------


@pytest.mark.usefixtures("waits_on")
async def test_polls_wait_a_few_seconds_at_a_time_for_a_minute(
    sleeps: list[float],
) -> None:
    sandbox = _make_scripted_sandbox(
        [_poll_wait_start_response(42), _RUNNING, _RUNNING, _poll_response()]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
    assert proc._server_wait_end is not None
    window = proc._server_wait_end - proc._started_at

    events = [event async for event in proc]

    assert events == [ExecCompleted(exit_code=0)]
    assert window == er.EXTENDED_WAIT_WINDOW_SECONDS
    assert [params["wait"] for params in _polls(sandbox)] == [
        er.EXTENDED_MAX_WAIT_SECONDS
    ] * 3
    assert sleeps == []


@pytest.mark.usefixtures("waits_off")
async def test_no_poll_waits_on_the_server_when_waits_are_off(
    sleeps: list[float],
) -> None:
    sandbox = _make_scripted_sandbox(
        [_poll_wait_start_response(42), _RUNNING, _RUNNING, _poll_response()]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    events = [event async for event in proc]

    assert events == [ExecCompleted(exit_code=0)]
    assert all("wait" not in params for params in _polls(sandbox))
    assert len(sleeps) == 2


@pytest.mark.usefixtures("waits_on")
async def test_a_stdin_open_process_still_gets_no_wait() -> None:
    sandbox = _make_scripted_sandbox([_poll_wait_start_response(42), _poll_response()])
    proc = await exec_remote_streaming(
        sandbox, ["cmd"], 5, er.ExecRemoteStreamingOptions(stdin_open=True)
    )

    _ = [event async for event in proc]

    assert all("wait" not in params for params in _polls(sandbox))


# --- the pause after a poll that did not wait ----------------------------------------


@pytest.mark.parametrize(
    ("age", "pause"), [(0.0, 0.25), (1.0, 1.0), (3.0, 3.0), (30.0, 5.0)]
)
def test_a_poll_that_did_not_wait_is_followed_by_a_pause_as_long_as_the_command_has_run(
    monkeypatch: pytest.MonkeyPatch, age: float, pause: float
) -> None:
    proc = ExecRemoteProcess(_mock_sandbox(), ["cmd"], ExecRemoteCommonOptions(), 5)
    proc._fork_polling = True
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    proc._started_at = 1000.0 - age

    assert proc._idle_wait() == pause


@pytest.mark.usefixtures("waits_off")
async def test_idle_polls_of_a_young_command_are_closer_together_than_the_interval(
    sleeps: list[float],
) -> None:
    sandbox = _make_scripted_sandbox(
        [_start_response(42), _RUNNING, _RUNNING, _poll_response()]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    _ = [event async for event in proc]

    assert len(sleeps) == 2
    assert all(er.POLL_BACKOFF_SECONDS <= pause < 5 for pause in sleeps)


@pytest.mark.usefixtures("waits_off")
async def test_a_poll_interval_the_caller_set_is_kept(sleeps: list[float]) -> None:
    sandbox = _make_scripted_sandbox(
        [_start_response(42), _RUNNING, _RUNNING, _poll_response()]
    )
    proc = await exec_remote_streaming(
        sandbox, ["cmd"], 5, ExecRemoteCommonOptions(poll_interval=7)
    )

    _ = [event async for event in proc]

    assert sleeps == [7, 7]


# --- the cap on waiting polls --------------------------------------------------------


@pytest.mark.usefixtures("waits_on")
async def test_a_poll_with_no_slot_to_wait_in_goes_out_unheld_and_is_paced(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.setenv(er.WAIT_CONCURRENCY_ENV, "1")
    sandbox = _make_scripted_sandbox(
        [_poll_wait_start_response(42), _RUNNING, _RUNNING, _poll_response()]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)
    slot = await get_or_create_semaphore("exec-remote-wait", 1, None, False)

    async with slot.semaphore:  # another command's poll is waiting
        events = [event async for event in proc]

    assert events == [ExecCompleted(exit_code=0)]
    assert all("wait" not in params for params in _polls(sandbox))
    # not polled back to back: each unheld poll is followed by a pause
    assert len(sleeps) == 2


@pytest.mark.usefixtures("waits_on")
async def test_a_waiting_poll_takes_a_slot_and_no_local_subprocess_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(er.WAIT_CONCURRENCY_ENV, "1")
    in_use: list[int] = []
    sandbox = _make_scripted_sandbox([_poll_wait_start_response(42), _poll_response()])
    scripted = sandbox.exec.side_effect

    async def record(*args: Any, **kwargs: Any) -> ExecResult[str]:
        if json.loads(kwargs["input"])["method"] == "exec_remote_poll":
            slot = await get_or_create_semaphore("exec-remote-wait", 1, None, False)
            in_use.append(slot.in_use)
        return await scripted(*args, **kwargs)

    sandbox.exec.side_effect = record
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    _ = [event async for event in proc]

    assert in_use == [1]
    start, poll = sandbox.exec.call_args_list
    assert start.kwargs["concurrency"] is True
    assert poll.kwargs["concurrency"] is False
    slot = await get_or_create_semaphore("exec-remote-wait", 1, None, False)
    assert slot.in_use == 0


# --- pacing an awaitable command's output --------------------------------------------


@pytest.mark.usefixtures("waits_on")
async def test_an_awaitable_command_that_keeps_printing_is_not_polled_back_to_back(
    sleeps: list[float],
) -> None:
    sandbox = _make_scripted_sandbox(
        [
            _poll_wait_start_response(42),
            _output("a"),
            _output("b"),
            _output("c"),
            _poll_response(stdout="d"),
        ]
    )

    result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

    assert result.stdout == "abcd"
    # none after the first output, whose command may be about to end
    assert sleeps == [er.OUTPUT_REPOLL_SECONDS, er.OUTPUT_REPOLL_SECONDS]


@pytest.mark.usefixtures("waits_on")
async def test_a_full_reply_is_followed_by_the_next_poll_at_once(
    sleeps: list[float],
) -> None:
    full = "x" * er.MAX_POLL_OUTPUT_BYTES
    sandbox = _make_scripted_sandbox(
        [_poll_wait_start_response(42), _output(full), _output(full), _poll_response()]
    )

    result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

    assert len(result.stdout) == 2 * er.MAX_POLL_OUTPUT_BYTES
    assert sleeps == []


@pytest.mark.usefixtures("waits_on")
async def test_a_streamed_command_gets_its_output_without_a_pause(
    sleeps: list[float],
) -> None:
    sandbox = _make_scripted_sandbox(
        [_poll_wait_start_response(42), _output("a"), _output("b"), _poll_response()]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    events = [event async for event in proc]

    assert events[:2] == [ExecStdout(data="a"), ExecStdout(data="b")]
    assert sleeps == []


# --- killing an awaitable command ----------------------------------------------------


def _answering(by_method: dict[str, str]) -> AsyncMock:
    """A sandbox that answers each request by its method, recording the methods."""
    sandbox = _mock_sandbox()
    sandbox.default_polling_interval.return_value = 5
    sandbox.no_events = _no_events_context
    sandbox.methods = []

    async def fake_exec(*args: Any, **kwargs: Any) -> ExecResult[str]:
        method = json.loads(kwargs["input"])["method"]
        sandbox.methods.append(method)
        if method == "exec_remote_poll":
            await anyio.sleep(0.01)
        return ExecResult(
            success=True, returncode=0, stdout=by_method[method], stderr=""
        )

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return sandbox


@pytest.mark.usefixtures("waits_on")
async def test_an_awaitable_command_that_times_out_is_killed_without_its_output() -> (
    None
):
    sandbox = _answering(
        {
            "exec_remote_start": _poll_wait_start_response(42),
            "exec_remote_poll": _RUNNING,
            "exec_remote_kill_key": _rpc({"killed": True}),
        }
    )

    with pytest.raises(TimeoutError):
        await exec_remote_awaitable(
            sandbox, ["sleep", "999"], 5, ExecRemoteAwaitableOptions(timeout=0.2)
        )

    assert "exec_remote_kill_key" in sandbox.methods
    assert "exec_remote_kill" not in sandbox.methods


@pytest.mark.usefixtures("waits_on")
async def test_a_streamed_command_is_still_killed_with_its_output() -> None:
    sandbox = _make_scripted_sandbox(
        [
            _poll_wait_start_response(42),
            _rpc({"seq": 1, "stdout": "last", "stderr": ""}),
        ]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    await proc.kill()

    assert [method for method, _ in sandbox.requests] == [
        "exec_remote_start",
        "exec_remote_kill",
    ]
    assert [event async for event in proc] == [ExecStdout(data="last")]


# --- an error the server reports for a poll ------------------------------------------


@pytest.mark.usefixtures("waits_off")
async def test_an_error_the_server_reports_for_a_poll_is_raised_without_a_retry() -> (
    None
):
    sandbox = _make_scripted_sandbox(
        [_start_response(42), _rpc_error("No job found with pid 42")]
    )
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    with pytest.raises(RuntimeError, match="No job found with pid 42"):
        _ = [event async for event in proc]

    assert len(_polls(sandbox)) == 1


# --- a final reply lost in transit ---------------------------------------------------


@pytest.mark.usefixtures("waits_off")
async def test_a_poll_whose_final_reply_was_lost_is_repeated_with_the_jobs_key() -> (
    None
):
    """The server answers a poll that names the job's key after the job ended."""
    sandbox = _make_scripted_sandbox(
        [
            _rpc({"pid": 42, "poll_replay": True}),
            RuntimeError("command terminated with exit code 137"),  # the reply, lost
            _poll_response(stdout="out", exit_code=3, seq=1),
        ]
    )

    with patch(
        "inspect_ai.util._sandbox.exec_remote.wait_exponential_jitter",
        new=lambda *a, **k: wait_none(),
    ):
        result = await exec_remote_awaitable(sandbox, ["cmd"], 5)

    assert (result.returncode, result.stdout) == (3, "out")
    start, lost, repeated = sandbox.requests
    assert repeated == lost
    assert lost[1]["job_key"] == start[1]["job_key"]


@pytest.mark.usefixtures("waits_off")
async def test_polls_name_no_key_to_a_server_that_does_not_take_one() -> None:
    sandbox = _make_scripted_sandbox([_start_response(42), _poll_response()])
    proc = await exec_remote_streaming(sandbox, ["cmd"], 5)

    _ = [event async for event in proc]

    assert all("job_key" not in params for params in _polls(sandbox))
