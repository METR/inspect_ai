from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

from ..._util.user_switch import RunAs

JobKey = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
"""Host-chosen job key: 32 lowercase hex digits (a UUID4 hex)."""


class SubmitParams(BaseModel):
    """Parameters for exec_remote_start."""

    command: str
    input: str | None = None
    """Standard input to send to the command (as a string)."""
    stdin_open: bool = False
    """If True, keep stdin open after writing initial input for later writes."""
    env: dict[str, str] | None = None
    """Additional environment variables (merged with the current environment)."""
    cwd: str | None = None
    """Working directory for command execution."""
    user: str | RunAs | None = None
    """User to run as: a username, or the sandbox default user's identity as
    captured by the host. Switching requires the server to run as root, unless
    the server already runs as that identity."""
    job_key: JobKey | None = None
    """Host-chosen key for the job, so the host can kill it with
    exec_remote_kill_key when the reply carrying its pid is lost."""
    model_config = {"extra": "forbid"}


class PollParams(BaseModel):
    """Parameters for exec_remote_poll."""

    pid: int
    ack_seq: int
    wait: float = Field(default=0, ge=0, le=60)
    """Seconds to wait for the job to exit or buffer output before replying."""
    model_config = {"extra": "forbid"}


MAX_WAIT_SECONDS = 5.0

# Longer waits are clamped, not rejected, so hosts need not know the server's limit.
# The wait RPCs ignore unknown fields, so a newer host can add optional ones; a
# field an older server must not ignore needs its own capability check.
WaitSeconds = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
    AfterValidator(lambda seconds: min(seconds, MAX_WAIT_SECONDS)),
]


class StartWaitParams(SubmitParams):
    """Start an awaitable command with a bounded initial completion wait."""

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    wait_seconds: WaitSeconds
    acks: list[str] = []
    """Request IDs whose terminal replies the host has received."""
    model_config = {"extra": "ignore"}


class PollWaitParams(BaseModel):
    """Wait for output or completion from a previously registered request."""

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    ack_seq: int = Field(ge=0)
    wait_seconds: WaitSeconds
    model_config = {"extra": "ignore"}


class KillWaitParams(BaseModel):
    """Cancel an awaitable command, including a pending launch."""

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    ack_seq: int = Field(ge=0)
    model_config = {"extra": "ignore"}


class KillParams(BaseModel):
    """Parameters for exec_remote_kill."""

    pid: int
    ack_seq: int
    model_config = {"extra": "forbid"}


class KillKeyParams(BaseModel):
    """Parameters for exec_remote_kill_key."""

    job_key: JobKey
    model_config = {"extra": "forbid"}


class KillKeyResult(BaseModel):
    """Result from exec_remote_kill_key.

    Fields:
        killed: True if a running job was killed; False if the job had already
            ended, or had not started (it is then refused when it arrives).
    """

    killed: bool


class SubmitResult(BaseModel):
    """Result from exec_remote_start."""

    pid: int
    poll_wait: bool = True
    """The server supports PollParams.wait."""


class PollResult(BaseModel):
    """Result from exec_remote_poll.

    Fields:
        state: Job lifecycle state - "running", "completed", or "killed"
        exit_code: Process exit code (only present when state is "completed")
        stdout: Standard output since last poll (incremental)
        stderr: Standard error since last poll (incremental)
    """

    state: Literal["running", "completed", "killed"]
    exit_code: int | None = None
    seq: int
    stdout: str
    stderr: str


class WaitResult(BaseModel):
    """Process identity and replayable output from an awaitable command."""

    pid: int
    result: PollResult


class OutputResult(BaseModel):
    """Base result containing incremental stdout/stderr since the last read.

    Used by kill, write_stdin, and close_stdin results so that every
    round-trip also delivers any buffered output.
    """

    seq: int
    stdout: str
    stderr: str


class KillResult(OutputResult):
    """Result from exec_remote_kill."""

    pass


class WriteStdinParams(BaseModel):
    """Parameters for exec_remote_write_stdin."""

    pid: int
    data: str
    ack_seq: int
    model_config = {"extra": "forbid"}


class WriteStdinResult(OutputResult):
    """Result from exec_remote_write_stdin."""

    pass


class CloseStdinParams(BaseModel):
    """Parameters for exec_remote_close_stdin."""

    pid: int
    ack_seq: int
    model_config = {"extra": "forbid"}


class CloseStdinResult(OutputResult):
    """Result from exec_remote_close_stdin."""

    pass
