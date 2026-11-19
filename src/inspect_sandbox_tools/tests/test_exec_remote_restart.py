"""METR fork: a server knows the exec_remote jobs an earlier server started.

The CLI starts a new server when it finds none running, and the host resends a
request that got no answer. Without a record on disk, a start resent after its
server died would run the command a second time.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from inspect_sandbox_tools._remote_tools._exec_remote._controller import Controller
from inspect_sandbox_tools._util.common_types import ToolException

KEY = "0123456789abcdef0123456789abcdef"
OTHER_KEY = "fedcba9876543210fedcba9876543210"


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    return tmp_path / "exec-remote-jobs"


@pytest.fixture
async def first(journal: Path) -> AsyncIterator[Controller]:
    """The server that dies: its jobs outlive it until the test ends."""
    controller = Controller()
    controller.persist_launches(journal)
    yield controller
    await controller.shutdown()


def restarted(journal: Path) -> Controller:
    """The server the CLI starts next, which has only the journal."""
    controller = Controller()
    controller.persist_launches(journal)
    return controller


async def test_a_start_resent_after_a_restart_does_not_run_the_command_again(
    first: Controller, journal: Path, tmp_path: Path
) -> None:
    ran = tmp_path / "ran"
    command = f"echo run >> {ran}; sleep 30"
    await first.submit(command, job_key=KEY)
    with anyio.fail_after(30):
        while not ran.exists():
            await anyio.sleep(0.01)
    second = restarted(journal)

    with pytest.raises(ToolException, match="may still be running"):
        await second.submit(command, job_key=KEY)

    assert ran.read_text() == "run\n"


async def test_a_job_an_earlier_server_started_is_reported_as_possibly_running(
    first: Controller, journal: Path
) -> None:
    pid = await first.submit("sleep 30", job_key=KEY)
    second = restarted(journal)

    with pytest.raises(ToolException, match="cannot kill it"):
        await second.kill_key(KEY)
    with pytest.raises(
        ToolException, match=f"No job found with pid {pid}: .* may still be running"
    ):
        await second.poll(pid, 0, job_key=KEY)
    # without the key it is any unknown PID, as before
    with pytest.raises(ToolException, match=f"No job found with pid {pid}$"):
        await second.poll(pid, 0)


async def test_a_key_killed_before_its_start_is_still_refused_after_a_restart(
    first: Controller, journal: Path
) -> None:
    await first.kill_key(KEY)
    second = restarted(journal)

    with pytest.raises(ToolException, match="was killed before it started"):
        await second.submit("true", job_key=KEY)
    assert (await second.kill_key(KEY)).killed is False


async def test_a_key_no_server_has_seen_starts_as_usual(
    first: Controller, journal: Path
) -> None:
    await first.submit("true", job_key=KEY)
    second = restarted(journal)

    pid = await second.submit("exit 3", job_key=OTHER_KEY)

    with anyio.fail_after(30):
        result = await second.poll(pid, 0, wait=30)
    assert (result.state, result.exit_code) == ("completed", 3)
    await second.shutdown()


async def test_a_record_cut_short_is_not_read_as_a_key_or_joined_to_the_next(
    journal: Path,
) -> None:
    journal.write_text("started 0123456789abcdef")  # its server died mid-write
    first = restarted(journal)
    await first.submit("true", job_key=OTHER_KEY)

    second = restarted(journal)

    assert second._restarted == {OTHER_KEY}
    await first.shutdown()


async def test_a_start_whose_key_cannot_be_recorded_is_refused(
    tmp_path: Path,
) -> None:
    ran = tmp_path / "ran"
    blocked = tmp_path / "exec-remote-jobs"
    blocked.mkdir()  # not a file that can be appended to
    controller = Controller()
    controller.persist_launches(tmp_path / "missing")
    controller._journal = blocked

    with pytest.raises(ToolException, match="could not be recorded as started"):
        await controller.submit(f"echo run >> {ran}", job_key=KEY)

    assert not ran.exists()
    assert controller._jobs == {}
