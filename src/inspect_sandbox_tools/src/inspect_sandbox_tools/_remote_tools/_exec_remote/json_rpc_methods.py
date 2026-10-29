import os

from ..._util.json_rpc_helpers import validated_json_rpc_method
from ._controller import Controller
from ._wait_controller import WaitController
from .tool_types import (
    CloseStdinParams,
    CloseStdinResult,
    KillParams,
    KillResult,
    KillWaitParams,
    PollParams,
    PollResult,
    PollWaitParams,
    StartWaitParams,
    SubmitParams,
    SubmitResult,
    WaitResult,
    WriteStdinParams,
    WriteStdinResult,
)

controller = Controller()
wait_controller = WaitController()
_can_switch_user = os.getuid() == 0


@validated_json_rpc_method(StartWaitParams)
async def exec_remote_start_wait(params: StartWaitParams) -> WaitResult:
    """Start an awaitable command and include its first result."""
    return await wait_controller.start(params, _can_switch_user)


@validated_json_rpc_method(PollWaitParams)
async def exec_remote_poll_wait(params: PollWaitParams) -> WaitResult:
    """Wait for output or completion of an awaitable command."""
    return await wait_controller.poll(
        params.request_id, params.ack_seq, params.wait_seconds
    )


@validated_json_rpc_method(KillWaitParams)
async def exec_remote_kill_wait(params: KillWaitParams) -> KillResult:
    """Cancel an awaitable command by its client-generated identity."""
    return await wait_controller.kill(params.request_id, params.ack_seq)


@validated_json_rpc_method(SubmitParams)
async def exec_remote_start(params: SubmitParams) -> SubmitResult:
    """Submit a command for async execution. Returns the PID."""
    pid = await controller.submit(
        params.command,
        input=params.input,
        stdin_open=params.stdin_open,
        env=params.env,
        cwd=params.cwd,
        user=params.user,
        can_switch_user=_can_switch_user,
    )
    return SubmitResult(pid=pid)


@validated_json_rpc_method(PollParams)
async def exec_remote_poll(params: PollParams) -> PollResult:
    """Poll job state and get incremental output."""
    return await controller.poll(params.pid, params.ack_seq)


@validated_json_rpc_method(KillParams)
async def exec_remote_kill(params: KillParams) -> KillResult:
    """Kill a running job."""
    return await controller.kill(params.pid, params.ack_seq)


@validated_json_rpc_method(WriteStdinParams)
async def exec_remote_write_stdin(params: WriteStdinParams) -> WriteStdinResult:
    """Write data to stdin of a running job."""
    return await controller.write_stdin(params.pid, params.data, params.ack_seq)


@validated_json_rpc_method(CloseStdinParams)
async def exec_remote_close_stdin(params: CloseStdinParams) -> CloseStdinResult:
    """Close stdin of a running job to signal EOF."""
    return await controller.close_stdin(params.pid, params.ack_seq)
