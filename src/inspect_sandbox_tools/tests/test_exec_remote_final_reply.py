"""METR fork: a poll repeated after a job ended gets the job's final reply again.

The reply that reports a job's end is the last one the host asks for. If it is
lost in transit the host repeats the poll, and a poll that names the job's key
is answered again instead of with "No job found".
"""

from collections.abc import AsyncIterator

import anyio
import pytest
from inspect_sandbox_tools._remote_tools._exec_remote import (
    _controller as controller_module,
)
from inspect_sandbox_tools._remote_tools._exec_remote._controller import Controller
from inspect_sandbox_tools._remote_tools._exec_remote.tool_types import PollResult
from inspect_sandbox_tools._util.common_types import ToolException

KEY = "0123456789abcdef0123456789abcdef"
OTHER_KEY = "fedcba9876543210fedcba9876543210"


@pytest.fixture
async def controller() -> AsyncIterator[Controller]:
    controller = Controller()
    yield controller
    await controller.shutdown()


async def _end(
    controller: Controller, command: str, key: str | None = KEY
) -> tuple[int, PollResult]:
    """Run `command` to its end; return its PID and the reply that reported the end."""
    pid = await controller.submit(command, job_key=key)
    with anyio.fail_after(30):
        while True:
            result = await controller.poll(pid, 0, wait=5, job_key=key)
            if result.state != "running":
                return pid, result


async def test_a_poll_repeated_after_the_end_gets_the_final_reply_again(
    controller: Controller,
) -> None:
    pid, final = await _end(controller, "printf out; printf err >&2; exit 3")

    again = await controller.poll(pid, 0, job_key=KEY)

    assert (final.state, final.exit_code) == ("completed", 3)
    assert (final.stdout, final.stderr) == ("out", "err")
    assert again == final


async def test_a_host_that_got_the_final_reply_is_sent_no_output_twice(
    controller: Controller,
) -> None:
    pid, final = await _end(controller, "printf out; exit 3")

    again = await controller.poll(pid, final.seq, job_key=KEY)

    assert (again.state, again.exit_code, again.stdout) == ("completed", 3, "")
    assert again.seq == final.seq


@pytest.mark.parametrize(
    ("started_with", "asked_with"),
    [(KEY, None), (KEY, OTHER_KEY), (None, None)],
    ids=["no key in the poll", "another job's key", "a job started without a key"],
)
async def test_a_repeated_poll_that_does_not_name_the_jobs_key_finds_no_job(
    controller: Controller, started_with: str | None, asked_with: str | None
) -> None:
    pid, _ = await _end(controller, "true", key=started_with)

    with pytest.raises(ToolException, match=f"No job found with pid {pid}"):
        await controller.poll(pid, 0, job_key=asked_with)


async def test_a_key_does_not_answer_for_another_pid(controller: Controller) -> None:
    pid, _ = await _end(controller, "true")

    with pytest.raises(ToolException, match="No job found"):
        await controller.poll(pid + 1, 0, job_key=KEY)


async def test_a_final_reply_is_forgotten_once_it_has_been_kept_long_enough(
    controller: Controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = await _end(controller, "printf out")
    job = controller._keyed_jobs[KEY]
    assert job.unacked_chars == 3

    monkeypatch.setattr(controller_module, "_FINAL_REPLY_SECONDS", 0.0)

    with pytest.raises(ToolException, match="No job found"):
        await controller.poll(pid, 0, job_key=KEY)
    assert job.unacked_chars == 0


async def test_the_oldest_final_replies_are_forgotten_when_too_much_is_kept(
    controller: Controller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller_module, "_FINAL_REPLY_CHARS", 10)
    first, _ = await _end(controller, "printf 12345678", key=KEY)
    second, kept = await _end(controller, "printf abcdefgh", key=OTHER_KEY)

    with pytest.raises(ToolException, match="No job found"):
        await controller.poll(first, 0, job_key=KEY)
    assert await controller.poll(second, 0, job_key=OTHER_KEY) == kept


async def test_a_killed_jobs_output_is_not_kept(controller: Controller) -> None:
    pid = await controller.submit("printf out; sleep 30", job_key=KEY)
    with anyio.fail_after(30):
        while controller._keyed_jobs[KEY].unacked_chars == 0:
            await controller.poll(pid, 0, wait=1, job_key=KEY)

    await controller.kill(pid, 0)

    assert controller._keyed_jobs[KEY].unacked_chars == 0
    with pytest.raises(ToolException, match="No job found"):
        await controller.poll(pid, 0, job_key=KEY)
