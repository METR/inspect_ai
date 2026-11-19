"""METR fork: a start that waits returns the first poll's result with the PID.

So a quick command takes the host one request, not a start and a poll.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from inspect_sandbox_tools._remote_tools._exec_remote import json_rpc_methods
from inspect_sandbox_tools._remote_tools._exec_remote._controller import Controller
from jsonrpcserver import async_dispatch

KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
async def controller(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Controller]:
    controller = Controller()
    monkeypatch.setattr(json_rpc_methods, "controller", controller)
    yield controller
    await controller.shutdown()


async def call(method: str, **params: Any) -> dict[str, Any]:
    request = {"jsonrpc": "2.0", "method": method, "id": 1, "params": params}
    reply = json.loads(await async_dispatch(json.dumps(request)))
    assert "error" not in reply, reply
    return reply["result"]


async def test_a_start_that_waits_returns_a_quick_commands_output_and_exit() -> None:
    started = await call(
        "exec_remote_start", command="printf out; exit 3", job_key=KEY, wait=30
    )

    assert started["result"] == {
        "state": "completed",
        "exit_code": 3,
        "seq": 1,
        "stdout": "out",
        "stderr": "",
    }


async def test_a_start_that_waits_returns_at_the_end_of_its_wait() -> None:
    began = time.monotonic()

    started = await call("exec_remote_start", command="sleep 30", job_key=KEY, wait=0.2)

    assert time.monotonic() - began < 5
    assert started["result"]["state"] == "running"
    # the command is polled, and ended, as any other
    polled = await call("exec_remote_poll", pid=started["pid"], ack_seq=1)
    assert polled["state"] == "running"
    await call("exec_remote_kill", pid=started["pid"], ack_seq=polled["seq"])


async def test_a_start_that_does_not_ask_to_wait_returns_no_result() -> None:
    started = await call("exec_remote_start", command="true", job_key=KEY)

    assert started["result"] is None
    assert started["start_wait"] is True


async def test_a_start_with_stdin_open_does_not_wait(controller: Controller) -> None:
    began = time.monotonic()

    started = await call(
        "exec_remote_start", command="cat", job_key=KEY, stdin_open=True, wait=30
    )

    assert time.monotonic() - began < 5
    assert started["result"] is None
    await controller.kill(started["pid"], 0)


async def test_a_command_that_ended_in_its_start_can_have_its_reply_again() -> None:
    started = await call(
        "exec_remote_start", command="printf out", job_key=KEY, wait=30
    )

    again = await call("exec_remote_poll", pid=started["pid"], ack_seq=0, job_key=KEY)

    assert again == started["result"]
