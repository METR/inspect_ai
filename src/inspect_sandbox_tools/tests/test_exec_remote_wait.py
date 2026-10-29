"""Awaitable exec tests own their controller and never start the shared server."""

import shlex
import shutil
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import pytest
from inspect_sandbox_tools._remote_tools._exec_remote import _job as job_module
from inspect_sandbox_tools._remote_tools._exec_remote import (
    _wait_controller as wait_module,
)
from inspect_sandbox_tools._remote_tools._exec_remote._controller import Controller
from inspect_sandbox_tools._remote_tools._exec_remote._job import Job, OutputChunk
from inspect_sandbox_tools._remote_tools._exec_remote._wait_controller import (
    WaitController,
)
from inspect_sandbox_tools._remote_tools._exec_remote.tool_types import (
    KillWaitParams,
    PollWaitParams,
    StartWaitParams,
    WaitResult,
)
from inspect_sandbox_tools._util.common_types import ToolException
from pydantic import ValidationError


def _python(code: str) -> str:
    return "exec " + shlex.join([sys.executable, "-c", code])


def _params(command: str, **kwargs: Any) -> StartWaitParams:
    return StartWaitParams(
        request_id=uuid4().hex, command=command, wait_seconds=1.0, **kwargs
    )


@pytest.fixture
async def controller() -> AsyncIterator[WaitController]:
    value = WaitController()
    try:
        yield value
    finally:
        with anyio.fail_after(10):
            await value.shutdown()


async def test_short_cat_returns_output_and_completion_together(
    controller: WaitController,
) -> None:
    expected = "A short file.\nCafé and 日本語.\n"
    params = _params("cat", input=expected)
    reply = await controller.start(params)

    assert reply.result.state == "completed"
    assert reply.result.exit_code == 0
    assert reply.result.stdout == expected
    assert reply.result.stderr == ""


async def test_start_preserves_stderr_and_nonzero_exit(
    controller: WaitController,
) -> None:
    params = _params("printf 'out'; printf 'err' >&2; exit 7")
    reply = await controller.start(params)

    assert reply.result.state == "completed"
    assert reply.result.exit_code == 7
    assert reply.result.stdout == "out"
    assert reply.result.stderr == "err"


async def test_lost_terminal_reply_is_replayed_without_relaunch(
    controller: WaitController, tmp_path: Path
) -> None:
    marker = tmp_path / "launches"
    params = _params(f"printf x >> {shlex.quote(str(marker))}; printf answer")
    first = await controller.start(params)
    replay = await controller.poll(params.request_id, 0, 0.0)
    restart = await controller.start(params)

    for reply in (replay, restart):
        assert reply.result.state == first.result.state == "completed"
        assert reply.result.stdout == first.result.stdout == "answer"
        assert reply.result.exit_code == first.result.exit_code == 0
        assert reply.pid == first.pid
    assert marker.read_text() == "x"


async def test_repeated_start_of_a_running_request_reports_it(
    controller: WaitController, tmp_path: Path
) -> None:
    marker = tmp_path / "launches"
    params = _params(f"printf x >> {shlex.quote(str(marker))}; sleep 60")
    params.wait_seconds = 0.3
    first = await controller.start(params)
    again = await controller.start(params)

    assert again.pid == first.pid
    assert again.result.state == "running"
    assert marker.read_text() == "x"


async def test_concurrent_duplicate_starts_launch_once(
    controller: WaitController, tmp_path: Path
) -> None:
    marker = tmp_path / "launches"
    params = _params(f"printf x >> {shlex.quote(str(marker))}; printf answer")
    replies: list[WaitResult] = []

    async def start() -> None:
        replies.append(await controller.start(params))

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(start)
        tasks.start_soon(start)

    assert [reply.result.stdout for reply in replies] == ["answer", "answer"]
    assert replies[0].pid == replies[1].pid
    assert marker.read_text() == "x"


async def test_start_returns_output_and_exit_together_within_the_wait(
    controller: WaitController,
) -> None:
    params = _params("printf first; sleep 0.3; printf second")
    reply = await controller.start(params)

    assert reply.result.state == "completed"
    assert reply.result.stdout == "firstsecond"


async def test_poll_wait_coalesces_output_until_the_deadline(
    controller: WaitController,
) -> None:
    params = _params(_python("import time; print('ready', flush=True); time.sleep(60)"))
    params.wait_seconds = 0.0
    first = await controller.start(params)
    began = time.monotonic()
    reply = await controller.poll(params.request_id, first.result.seq, 0.5)

    assert time.monotonic() - began >= 0.45
    assert reply.result.state == "running"
    assert first.result.stdout + reply.result.stdout == "ready\n"


async def test_poll_wait_returns_once_a_reply_is_full(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_module, "_MAX_POLL_OUTPUT_BYTES", 8)
    params = _params(
        _python("import time; print('x' * 16, flush=True); time.sleep(60)")
    )
    params.wait_seconds = 0.0
    first = await controller.start(params)
    with anyio.fail_after(2):
        reply = await controller.poll(params.request_id, first.result.seq, 5.0)

    assert reply.result.state == "running"
    assert first.result.stdout + reply.result.stdout == "x" * 8


async def test_output_from_a_child_after_exit_is_kept(
    controller: WaitController,
) -> None:
    params = _params("(sleep 0.3; printf late) & printf early")
    reply = await controller.start(params)

    assert reply.result.state == "completed"
    assert reply.result.stdout == "earlylate"


async def test_a_child_flooding_the_pipe_does_not_block_completion(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_module, "_READER_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(job_module, "_BACKPRESSURE_BUFFER_SIZE", 4 * 1024 * 1024)
    params = _params("yes & sleep 0.3; exit 3")  # the pipe is full before exit
    reply = await controller.start(params)
    with anyio.fail_after(10):
        while reply.result.state == "running":
            assert len(reply.result.stdout) <= job_module._MAX_POLL_OUTPUT_BYTES
            await anyio.sleep(0.05)  # a host round trip, while the child refills
            reply = await controller.poll(params.request_id, reply.result.seq, 5.0)

    assert reply.result.state == "completed"
    assert reply.result.exit_code == 3


async def test_running_output_is_replayed_until_acknowledged(
    controller: WaitController,
) -> None:
    params = _params(_python("import time; print('ready', flush=True); time.sleep(60)"))
    params.wait_seconds = 0.0
    first = await controller.start(params)
    acknowledged_seq = 0 if first.result.stdout else first.result.seq
    output = (
        first
        if first.result.stdout
        else await controller.poll(params.request_id, acknowledged_seq, 0.5)
    )
    replay = await controller.poll(params.request_id, acknowledged_seq, 0.0)
    acknowledged = await controller.poll(params.request_id, output.result.seq, 0.0)

    assert output.result.stdout == replay.result.stdout == "ready\n"
    assert acknowledged.result.stdout == ""
    assert acknowledged.result.state == "running"


async def test_large_unicode_output_is_delivered_exactly_once(
    controller: WaitController,
) -> None:
    expected = "€日本語🙂" * 200_000
    params = _params(_python("import sys; sys.stdout.write('€日本語🙂' * 200000)"))
    reply = await controller.start(params)
    chunks: list[str] = []
    with anyio.fail_after(10):
        while True:
            chunks.append(reply.result.stdout)
            assert reply.result.stderr == ""
            if reply.result.state == "completed":
                break
            reply = await controller.poll(params.request_id, reply.result.seq, 5.0)

    assert reply.result.exit_code == 0
    assert "".join(chunks) == expected


async def test_cancelled_start_keeps_launch_owned_and_recoverable(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    done = anyio.Event()
    original_create = Job.create
    calls = 0

    async def delayed_create(*args: Any, **kwargs: Any) -> Job:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(Job, "create", delayed_create)
    params = _params("printf recovered")
    scopes: list[anyio.CancelScope] = []

    async def start() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await controller.start(params)
            finally:
                done.set()

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(start)
                await entered.wait()
                scopes[0].cancel()
                await done.wait()
                release.set()
                recovered = await controller.poll(params.request_id, 0, 5.0)
                output = recovered.result.stdout
                while recovered.result.state != "completed":
                    recovered = await controller.poll(
                        params.request_id, recovered.result.seq, 5.0
                    )
                    output += recovered.result.stdout
    finally:
        release.set()

    assert calls == 1
    assert recovered.result.state == "completed"
    assert output == "recovered"


async def test_kill_while_launch_is_pending_cleans_up_the_created_process(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    killing = anyio.Event()
    original_create = Job.create
    jobs: list[Job] = []

    async def delayed_create(*args: Any, **kwargs: Any) -> Job:
        entered.set()
        await release.wait()
        job = await original_create(*args, **kwargs)
        jobs.append(job)
        return job

    monkeypatch.setattr(Job, "create", delayed_create)
    params = _params("sleep 60")

    async def start() -> None:
        await controller.start(params)

    async def kill() -> None:
        killing.set()
        await controller.kill(params.request_id, 0)

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(start)
                await entered.wait()
                tasks.start_soon(kill)
                await killing.wait()
                release.set()
    finally:
        release.set()

    assert all(job._process.returncode is not None for job in jobs)


async def test_launch_error_is_raised_as_a_legacy_start_raises_it(
    controller: WaitController, tmp_path: Path
) -> None:
    missing = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError) as legacy:
        await Controller().submit("true", cwd=missing)
    with pytest.raises(FileNotFoundError) as waited:
        await controller.start(_params("true", cwd=missing))

    assert repr(waited.value) == repr(legacy.value)


async def test_a_lost_result_says_how_the_command_ended(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = _params("true", cwd=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        await controller.start(failed)
    killed = _params("sleep 60")
    killed.wait_seconds = 0.0
    await controller.start(killed)
    await controller.kill(killed.request_id, 0)
    monkeypatch.setattr(wait_module, "_TERMINAL_TTL", 0.0)
    controller._prune()

    with pytest.raises(ToolException, match=r"could not be started \(FileNotFound"):
        await controller.start(failed)
    with pytest.raises(
        ToolException, match="result lost: the command ran and was killed"
    ):
        await controller.start(killed)


async def test_kill_before_start_prevents_a_late_launch(
    controller: WaitController, tmp_path: Path
) -> None:
    marker = tmp_path / "should-not-exist"
    params = _params(f"printf unexpected > {shlex.quote(str(marker))}")

    await controller.kill(params.request_id, 0)
    with pytest.raises(ToolException, match="cancelled before it started"):
        await controller.start(params)

    assert not marker.exists()


async def test_kill_interrupts_initial_input_to_a_nonreading_process(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = anyio.Event()
    original_create = Job.create
    jobs: list[Job] = []

    async def capture_create(*args: Any, **kwargs: Any) -> Job:
        register = kwargs.pop("on_create")

        def on_create(job: Job) -> None:
            jobs.append(job)
            register(job)
            spawned.set()

        kwargs["on_create"] = on_create
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(Job, "create", capture_create)
    params = _params(_python("import time; time.sleep(60)"), input="x" * 1_000_000)

    async def start() -> None:
        await controller.start(params)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(start)
            await spawned.wait()
            await controller.kill(params.request_id, 0)

    assert len(jobs) == 1
    assert jobs[0]._process.returncode is not None


@pytest.mark.parametrize(
    "setsid",
    [
        "",
        pytest.param(
            "setsid ",
            marks=pytest.mark.skipif(
                shutil.which("setsid") is None, reason="needs util-linux setsid"
            ),
        ),
    ],
)
async def test_shutdown_ends_a_launch_blocked_by_a_child_holding_stdin(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch, setsid: str
) -> None:
    """The parent exits; its child (in or out of its process group) keeps stdin open."""
    spawned = anyio.Event()
    original_create = Job.create
    jobs: list[Job] = []

    async def capture_create(*args: Any, **kwargs: Any) -> Job:
        register = kwargs.pop("on_create")

        def on_create(job: Job) -> None:
            jobs.append(job)
            register(job)
            spawned.set()

        kwargs["on_create"] = on_create
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(Job, "create", capture_create)
    params = _params(f"exec 3<&0; {setsid}sleep 60 <&3 & exit 0", input="x" * 1_000_000)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(controller.start, params)
        await spawned.wait()
        with anyio.fail_after(5):
            while jobs[0]._process.returncode is None:
                await anyio.sleep(0.05)
        with anyio.fail_after(10):
            await controller.shutdown()


async def test_cancelled_failed_launch_is_prunable(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    killing = anyio.Event()

    async def failed_create(*args: Any, **kwargs: Any) -> Job:
        entered.set()
        await release.wait()
        raise OSError("launch failed")

    monkeypatch.setattr(Job, "create", failed_create)
    params = _params("true")

    async def start() -> None:
        with pytest.raises(ToolException):
            await controller.start(params)

    async def kill() -> None:
        killing.set()
        await controller.kill(params.request_id, 0)

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(start)
            await entered.wait()
            tasks.start_soon(kill)
            await killing.wait()
            release.set()

    monkeypatch.setattr(wait_module, "_TERMINAL_TTL", 0.0)
    controller._prune()
    assert params.request_id not in controller._requests


async def test_killed_output_is_replayed_after_a_lost_kill_reply(
    controller: WaitController,
) -> None:
    params = _params(_python("import time; print('ready', flush=True); time.sleep(60)"))
    first = await controller.start(params)
    assert first.result.stdout == "ready\n"

    killed = await controller.kill(params.request_id, 0)
    replay = await controller.poll(params.request_id, 0, 0.0)
    acknowledged = await controller.poll(params.request_id, replay.result.seq, 0.0)

    assert replay.result.state == "killed"
    assert killed.stdout == replay.result.stdout == "ready\n"
    assert acknowledged.result.stdout == ""
    assert acknowledged.result.state == "killed"


async def test_kill_after_completion_keeps_the_completed_reply(
    controller: WaitController,
) -> None:
    params = _params("printf done; exit 3")
    assert (await controller.start(params)).result.state == "completed"

    killed = await controller.kill(params.request_id, 0)
    replay = await controller.poll(params.request_id, 0, 0.0)

    assert killed.stdout == "done"
    assert replay.result.state == "completed"
    assert replay.result.exit_code == 3
    assert replay.result.stdout == "done"


async def test_kill_wakes_an_inflight_long_poll(controller: WaitController) -> None:
    params = _params("sleep 60")
    params.wait_seconds = 0.0
    first = await controller.start(params)
    polling = anyio.Event()
    replies: list[WaitResult] = []

    async def poll() -> None:
        polling.set()
        replies.append(await controller.poll(params.request_id, first.result.seq, 5.0))

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(poll)
            await polling.wait()
            await controller.kill(params.request_id, first.result.seq)

    assert replies[0].result.state == "killed"


async def test_cancelled_kill_does_not_retain_completed_output_task(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    returned = anyio.Event()
    original_kill = Job.kill

    async def delayed_kill(
        job: Job, ack_seq: int, timeout: int = 5, *, drain_terminal: bool = False
    ) -> OutputChunk:
        entered.set()
        await release.wait()
        return await original_kill(job, ack_seq, timeout, drain_terminal=drain_terminal)

    monkeypatch.setattr(Job, "kill", delayed_kill)
    params = _params(_python("import time; print('ready', flush=True); time.sleep(60)"))
    await controller.start(params)
    scopes: list[anyio.CancelScope] = []

    async def kill() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await controller.kill(params.request_id, 0)
            finally:
                returned.set()

    try:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(kill)
                await entered.wait()
                scopes[0].cancel()
                await returned.wait()
                entry = controller._requests[params.request_id]
                stop = entry.stop
                assert stop is not None
                release.set()
                result = await stop
                await anyio.sleep(0)
    finally:
        release.set()

    assert result.stdout == "ready\n"
    assert entry.stop is None
    acknowledged = await controller.poll(params.request_id, result.seq, 0.0)
    assert acknowledged.result.stdout == ""


@pytest.mark.parametrize("launched", [False, True])
async def test_repeated_kill_does_not_extend_terminal_ttl(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch, launched: bool
) -> None:
    params = _params("sleep 60")
    params.wait_seconds = 0.0
    if launched:
        await controller.start(params)
    clock = [10.0]
    monkeypatch.setattr(wait_module, "monotonic", lambda: clock[0])
    await controller.kill(params.request_id, 0)
    clock[0] = 200.0
    await controller.kill(params.request_id, 0)
    clock[0] = 310.1
    controller._prune()

    assert params.request_id not in controller._requests


async def test_terminal_replay_expires(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _params("printf done")
    reply = await controller.start(params)
    assert reply.result.state == "completed"

    monkeypatch.setattr(wait_module, "_TERMINAL_TTL", 0.0)
    controller._prune()

    with pytest.raises(ToolException):
        await controller.poll(params.request_id, 0, 0.0)


async def test_start_after_the_reply_expired_does_not_relaunch(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "launches"
    params = _params(f"printf x >> {shlex.quote(str(marker))}")
    assert (await controller.start(params)).result.state == "completed"
    monkeypatch.setattr(wait_module, "_TERMINAL_TTL", 0.0)
    controller._prune()

    with pytest.raises(
        ToolException, match="result lost: the command ran and exited with code 0"
    ):
        await controller.start(params)
    with pytest.raises(
        ToolException, match="result lost: the command ran and exited with code 0"
    ):
        await controller.poll(params.request_id, 0, 0.0)
    assert marker.read_text() == "x"


async def test_expired_ids_are_forgotten_beyond_their_bounds(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    monkeypatch.setattr(wait_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(wait_module, "_TERMINAL_TTL", 0.0)
    monkeypatch.setattr(wait_module, "_MAX_EXPIRED_IDS", 2)
    requests = [_params("true") for _ in range(3)]
    for params in requests:
        await controller.start(params)

    assert list(controller._expired) == [params.request_id for params in requests[1:]]
    clock[0] = wait_module._EXPIRED_ID_TTL
    controller._prune()
    assert not controller._expired


async def test_unacknowledged_replies_survive_other_output(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wait_module, "_MAX_ACKNOWLEDGED_ENTRIES", 0)
    output = "x" * 512 * 1024
    requests = [
        _params(_python(f"print('x' * {len(output)}, end='')")) for _ in range(5)
    ]
    for params in requests:
        reply = await controller.start(params)
        assert reply.result.state == "completed"
    controller._prune()

    for params in requests:
        replay = await controller.poll(params.request_id, 0, 0.0)
        assert replay.result.stdout == output


async def test_retained_replies_are_bounded_in_total(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = "x" * 4096
    reply_bytes = sys.getsizeof(output) + sys.getsizeof("")
    monkeypatch.setattr(wait_module, "_MAX_RETAINED_BYTES", 3 * reply_bytes)
    requests = [
        _params(_python(f"print('x' * {len(output)}, end='')")) for _ in range(4)
    ]
    for params in requests:
        assert (await controller.start(params)).result.state == "completed"

    with pytest.raises(
        ToolException, match="result lost: the command ran and exited with code 0"
    ):
        await controller.poll(requests[0].request_id, 0, 0.0)
    for params in requests[1:]:
        assert (
            await controller.poll(params.request_id, 0, 0.0)
        ).result.stdout == output


async def test_kill_reply_holds_at_most_one_reply_of_output(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job_module, "_MAX_POLL_OUTPUT_BYTES", 8)
    params = _params(
        _python("import time; print('x' * 64, end='', flush=True); time.sleep(60)")
    )
    params.wait_seconds = 0.0
    first = await controller.start(params)
    entry = controller._requests[params.request_id]
    job = entry.job
    assert job is not None
    with anyio.fail_after(5):
        while len(first.result.stdout) + job._stdout_buffer.size < 64:
            await anyio.sleep(0.05)
    # a full reply that never reaches the host, so its kill acknowledges `first`
    lost = await controller.poll(params.request_id, first.result.seq, 0.0)
    assert lost.result.stdout == "x" * 8

    killed = await controller.kill(params.request_id, first.result.seq)

    assert killed.stdout == "x" * 8
    assert job._stdout_buffer.size == 0
    # the stored kill reply is the only copy of its output
    assert job.held_output_bytes == 0
    assert wait_module._retained_bytes(entry) == sys.getsizeof(
        killed.stdout
    ) + sys.getsizeof(killed.stderr)


async def test_an_exited_request_waits_for_a_late_poll(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the host's wait window it polls every `poll_interval`, however long."""
    clock = [0.0]
    monkeypatch.setattr(wait_module, "monotonic", lambda: clock[0])
    params = _params("sleep 0.2; printf done")
    params.wait_seconds = 0.0
    first = await controller.start(params)
    assert first.result.state == "running"
    job = controller._requests[params.request_id].job
    assert job is not None
    with anyio.fail_after(5):
        while job._process.returncode is None:
            await anyio.sleep(0.05)

    clock[0] = 10 * wait_module._TERMINAL_TTL
    controller._prune()
    final = await controller.poll(params.request_id, first.result.seq, 0.0)

    assert (final.result.state, final.result.stdout) == ("completed", "done")


async def test_an_unsent_result_is_dropped_only_after_sent_replies(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _params(_python("print('x' * 4096, end='')"))
    assert (await controller.start(sent)).result.state == "completed"
    unsent = _params("sleep 0.2; printf done")
    unsent.wait_seconds = 0.0
    first = await controller.start(unsent)
    entry = controller._requests[unsent.request_id]
    assert entry.job is not None
    with anyio.fail_after(5):
        while entry.job._process.returncode is None:
            await anyio.sleep(0.05)

    # room for the unsent result only
    retained = wait_module._retained_bytes(entry)
    monkeypatch.setattr(wait_module, "_MAX_RETAINED_BYTES", retained)
    controller._prune()

    assert sent.request_id in controller._expired
    final = await controller.poll(unsent.request_id, first.result.seq, 0.0)
    assert (final.result.state, final.result.stdout) == ("completed", "done")


async def test_an_unsent_result_over_the_byte_cap_is_reported_lost(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    params = _params("sleep 0.2")
    params.wait_seconds = 0.0
    first = await controller.start(params)
    job = controller._requests[params.request_id].job
    assert job is not None
    with anyio.fail_after(5):
        while job._process.returncode is None:
            await anyio.sleep(0.05)

    monkeypatch.setattr(wait_module, "_MAX_RETAINED_BYTES", 0)
    controller._prune()

    with pytest.raises(ToolException, match="exited, but its final reply was dropped"):
        await controller.poll(params.request_id, first.result.seq, 0.0)


async def test_a_poll_waiting_when_its_unsent_result_is_dropped_reports_it_lost(
    controller: WaitController,
) -> None:
    params = _params("printf done; sleep 0.2")
    params.wait_seconds = 0.0
    first = await controller.start(params)
    entry = controller._requests[params.request_id]
    assert entry.job is not None
    with anyio.fail_after(5):
        while entry.job._process.returncode is None:
            await anyio.sleep(0.05)

    # as if the byte cap dropped it between the poll's lookup and its reply
    controller._evict(params.request_id, entry)

    with pytest.raises(ToolException, match="exited, but its final reply was dropped"):
        await controller._poll(entry, params.request_id, first.result.seq)


async def test_an_unsent_result_is_kept_while_its_poll_settles_the_readers(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The byte cap cannot release the output of a reply being made."""
    monkeypatch.setattr(job_module, "_READER_GRACE_SECONDS", 60.0)
    settling = anyio.Event()
    wait_for_readers = Job._wait_for_readers

    async def signal_then_wait(job: Job) -> None:
        settling.set()
        await wait_for_readers(job)

    monkeypatch.setattr(Job, "_wait_for_readers", signal_then_wait)
    go, release = tmp_path / "go", tmp_path / "release"

    def wait_for(path: Path) -> str:
        return f"while [ ! -e {shlex.quote(str(path))} ]; do sleep 0.01; done"

    # the leader prints and exits once `go` exists; a child holds the pipes until
    # `release` exists
    params = _params(f"{wait_for(go)}; printf done; ({wait_for(release)}) &")
    params.wait_seconds = 0.0
    first = await controller.start(params)
    assert first.result.state == "running"
    job = controller._requests[params.request_id].job
    assert job is not None
    go.touch()
    with anyio.fail_after(5):
        while not job.exited or job._stdout_buffer.size == 0:
            await anyio.sleep(0.05)

    replies: list[WaitResult] = []

    async def poll() -> None:
        replies.append(await controller.poll(params.request_id, first.result.seq, 0.0))

    try:
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(poll)
                await settling.wait()
                # as a start or poll of another request would
                monkeypatch.setattr(wait_module, "_MAX_RETAINED_BYTES", 0)
                controller._prune()
                release.touch()
    finally:
        release.touch()

    assert (replies[0].result.state, replies[0].result.stdout) == ("completed", "done")


async def test_acknowledged_reply_keeps_only_a_marker(
    controller: WaitController, tmp_path: Path
) -> None:
    marker = tmp_path / "launches"
    params = _params(f"printf x >> {shlex.quote(str(marker))}; printf answer")
    first = await controller.start(params)
    assert first.result.stdout == "answer"

    await controller.start(_params("true", acks=[params.request_id]))
    restart = await controller.start(params)

    assert restart.pid == first.pid
    assert restart.result.state == "completed"
    assert restart.result.stdout == ""
    assert marker.read_text() == "x"


async def test_acknowledged_entry_count_is_bounded(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wait_module, "_MAX_ACKNOWLEDGED_ENTRIES", 1)
    requests = [_params("printf done") for _ in range(3)]
    for params in requests:
        reply = await controller.start(params)
        assert reply.result.state == "completed"
    await controller.start(
        _params("true", acks=[params.request_id for params in requests[:2]])
    )

    with pytest.raises(ToolException):
        await controller.poll(requests[0].request_id, 0, 0.0)
    acknowledged = await controller.poll(requests[1].request_id, 0, 0.0)
    assert acknowledged.result.stdout == ""
    unacknowledged = await controller.poll(requests[2].request_id, 0, 0.0)
    assert unacknowledged.result.stdout == "done"


async def test_retention_limits_do_not_evict_running_jobs(
    controller: WaitController, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wait_module, "_MAX_ACKNOWLEDGED_ENTRIES", 0)
    running = _params("sleep 60")
    running.wait_seconds = 0.0
    started = await controller.start(running)
    for _ in range(2):
        completed = await controller.start(_params("true", acks=[running.request_id]))
        assert completed.result.state == "completed"
    controller._prune()

    remaining = await controller.poll(running.request_id, started.result.seq, 0.0)
    assert remaining.pid == started.pid
    assert remaining.result.state == "running"


def test_long_wait_duration_is_clamped() -> None:
    start = StartWaitParams(request_id=uuid4().hex, command="true", wait_seconds=60)
    poll = PollWaitParams(request_id=uuid4().hex, ack_seq=0, wait_seconds=5.1)

    assert start.wait_seconds == poll.wait_seconds == 5.0


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_invalid_wait_duration_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        StartWaitParams(request_id=uuid4().hex, command="true", wait_seconds=value)
    with pytest.raises(ValidationError):
        PollWaitParams(request_id=uuid4().hex, ack_seq=0, wait_seconds=value)


def test_wait_params_ignore_fields_from_a_newer_host() -> None:
    request_id = uuid4().hex
    newer = {"acks": [uuid4().hex], "added_later": {"x": 1}}

    start = StartWaitParams.model_validate(
        {"request_id": request_id, "command": "true", "wait_seconds": 1, **newer}
    )
    poll = PollWaitParams.model_validate(
        {"request_id": request_id, "ack_seq": 2, "wait_seconds": 1, **newer}
    )
    kill = KillWaitParams.model_validate(
        {"request_id": request_id, "ack_seq": 2, **newer}
    )

    assert (start.request_id, start.command) == (request_id, "true")
    assert (poll.ack_seq, kill.ack_seq) == (2, 2)
    assert (
        "added_later" not in start.model_dump() | poll.model_dump() | kill.model_dump()
    )
