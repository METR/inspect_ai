import asyncio
import re
import time
from pathlib import Path

from inspect_sandbox_tools._util.common_types import ToolException
from inspect_sandbox_tools._util.server_dir import (
    open_private_append,
    read_private_text,
)
from inspect_sandbox_tools._util.user_switch import RunAs

from ._job import Job
from .tool_types import (
    CloseStdinResult,
    KillKeyResult,
    KillResult,
    PollResult,
    WriteStdinResult,
)

# METR fork: how long, and how much of, the final replies of ended jobs are kept
# for a host whose poll lost one (see Controller.poll).
_FINAL_REPLY_SECONDS = 300.0
_FINAL_REPLY_CHARS = 32 * 1024 * 1024

# METR fork: what is said of a job an earlier server started (see
# Controller.persist_launches).
_RESTARTED = (
    "was started before the sandbox tools server restarted, and may still be running"
)
_JOB_KEY = re.compile(r"[0-9a-f]{32}")


class Controller:
    """Simple job registry keyed by PID.

    Unlike bash_session's SessionController, exec_remote uses PIDs as natural
    unique identifiers - no session naming or multiplexing needed.

    A job may also carry a host-chosen key, so the host can kill it when the
    reply carrying its PID never arrives (see `kill_key`).
    """

    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._retired_jobs: list[Job] = []
        # Jobs started with a key. Entries outlive the job (as _retired_jobs
        # does) so that a repeated start with the same key never starts a
        # second job.
        self._keyed_jobs: dict[str, Job] = {}
        # Starts in progress, by key: resolves to the job once its process
        # exists, or None if the spawn failed.
        self._spawning: dict[str, asyncio.Future[Job | None]] = {}
        # Keys killed before a start with them spawned a process; a start with
        # one is refused, including a retry after the killed start's spawn failed.
        self._refused_keys: set[str] = set()
        # METR fork: the keys of running jobs, by PID, and the ended jobs whose
        # final reply can still be sent again, oldest first, by key, with when
        # they ended.
        self._running_keys: dict[int, str] = {}
        self._final_replies: dict[str, tuple[Job, float]] = {}
        # METR fork: where started and refused keys are recorded, and the keys an
        # earlier server recorded as started (see persist_launches).
        self._journal: Path | None = None
        self._restarted: set[str] = set()

    def persist_launches(self, path: Path) -> None:
        """Record the keys of the jobs this server starts in `path`, and learn an earlier server's.

        The CLI starts a new server when it finds none running, and the host
        resends a request that got no answer. So when a server dies with a start
        in flight, the resent start reaches a server that knows nothing of the
        first and would run the command a second time. With the keys on disk,
        a server refuses to start a key an earlier one started, says of it that
        the command may still be running where it would have said "no job", and
        still refuses a key that was killed before its start arrived.

        A key is recorded before its process is started, and a start whose key
        cannot be recorded is refused.
        """
        try:
            records = read_private_text(path).splitlines()
        except FileNotFoundError:
            records = []
        for record in records:
            action, _, key = record.partition(" ")
            if not _JOB_KEY.fullmatch(key):
                continue  # a record cut short by the end of its server
            if action == "started":
                self._restarted.add(key)
            elif action == "refused":
                self._refused_keys.add(key)
        self._journal = path

    def _record(self, action: str, job_key: str) -> None:
        if self._journal is None:
            return
        try:
            with open_private_append(self._journal) as journal:
                # On a line of its own: a record cut short must not join this one.
                journal.write(f"\n{action} {job_key}\n")
        except (OSError, RuntimeError) as ex:  # RuntimeError: not a file of ours
            raise ToolException(
                f"exec_remote job {job_key} could not be recorded as {action}: {ex}"
            ) from ex

    async def submit(
        self,
        command: str,
        input: str | None = None,
        stdin_open: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        user: str | RunAs | None = None,
        can_switch_user: bool = False,
        job_key: str | None = None,
    ) -> int:
        """Create a new job and return its PID.

        The job is registered as soon as its process exists, before its initial
        input is written, so `kill_key` and server shutdown reach it while the
        start waits on stdin. If writing the input fails or is cancelled, the
        job is killed before the error propagates.

        A start whose key was killed first is refused. A repeated start with
        the key of a running job (a retried request) returns that job's PID;
        one whose job is still starting or has ended is refused.

        Args:
            command: The shell command to execute.
            input: Optional standard input to send to the command.
            stdin_open: If True, keep stdin open for later writes.
            env: Additional environment variables (merged with current env).
            cwd: Working directory for command execution.
            user: User to run the command as (requires can_switch_user=True).
            can_switch_user: Whether the server can switch users (running as root).
            job_key: Host-chosen key that `kill_key` accepts.

        Raises:
            ToolException: If the job key was killed or has already been used.
        """
        if job_key is not None:
            if job_key in self._refused_keys:
                raise ToolException(
                    f"exec_remote job {job_key} was killed before it started"
                )
            if job_key in self._restarted:
                raise ToolException(
                    f"exec_remote job {job_key} {_RESTARTED}; it is not started again"
                )
            if job_key in self._spawning:
                raise ToolException(f"exec_remote job {job_key} is already starting")
            if job_key in self._keyed_jobs:
                return self._repeated_start(job_key)
        spawned: asyncio.Future[Job | None] = asyncio.get_running_loop().create_future()
        if job_key is not None:
            self._spawning[job_key] = spawned
        try:
            try:
                if job_key is not None:
                    self._record("started", job_key)
                job = await Job.spawn(
                    command,
                    stdin=input is not None or stdin_open,
                    env=env,
                    cwd=cwd,
                    user=user,
                    can_switch_user=can_switch_user,
                )
            except BaseException:
                spawned.set_result(None)
                raise
            self._jobs[job.pid] = job
            if job_key is not None:
                self._keyed_jobs[job_key] = job
                self._running_keys[job.pid] = job_key
            spawned.set_result(job)
            await self._initialize(job, input=input, stdin_open=stdin_open)
            return job.pid
        finally:
            if job_key is not None:
                del self._spawning[job_key]

    async def _initialize(
        self, job: Job, *, input: str | None, stdin_open: bool
    ) -> None:
        """Initialize a registered job's stdin, killing the job if that fails."""
        try:
            await job.initialize_stdin(input, stdin_open)
        except BaseException:
            if self._jobs.get(job.pid) is job:
                await self.kill(job.pid, ack_seq=0)
            raise

    def _repeated_start(self, job_key: str) -> int:
        job = self._keyed_jobs[job_key]
        if self._jobs.get(job.pid) is not job:
            raise ToolException(f"exec_remote job {job_key} has already ended")
        return job.pid

    async def kill_key(self, job_key: str) -> KillKeyResult:
        """Kill the job started with `job_key`, or refuse its start if none has.

        The host calls this when it submitted a start but never received the
        PID. A key with no spawned process yet is refused, so a later start
        with it (a delayed request, or a retry after a failed spawn) is not run.
        For a start in progress, this waits until its process exists (or its
        spawn fails), then kills it, which also ends any stdin write the start
        is waiting on. It returns once the job has terminated.
        """
        if job_key in self._restarted:
            raise ToolException(
                f"exec_remote job {job_key} {_RESTARTED}; this server cannot kill it"
            )
        if job_key not in self._keyed_jobs and job_key not in self._refused_keys:
            # Refused from here on even if it cannot be recorded, which is raised:
            # a later server would then not know to refuse the start.
            self._refused_keys.add(job_key)
            self._record("refused", job_key)
        spawned = self._spawning.get(job_key)
        if spawned is not None:
            job = await asyncio.shield(spawned)
        else:
            job = self._keyed_jobs.get(job_key)
        if job is None:
            return KillKeyResult(killed=False)
        if self._jobs.get(job.pid) is not job:
            await job.await_termination()
            return KillKeyResult(killed=False)
        await self.kill(job.pid, ack_seq=0)
        return KillKeyResult(killed=job.killed)

    async def poll(
        self, pid: int, ack_seq: int, wait: float = 0, job_key: str | None = None
    ) -> PollResult:
        """Get job state and incremental output. Auto-cleanup on terminal state.

        The reply that reports a job's end is the last the host asks for, so if
        it is lost the host repeats the poll for a job that is gone. A poll
        that names the job's key (METR fork) is then answered with that reply
        again, for `_FINAL_REPLY_SECONDS` after the end and while the replies
        kept are within `_FINAL_REPLY_CHARS`.
        """
        job = self._jobs.get(pid)
        if job is None:
            return self._final_reply(pid, ack_seq, job_key)
        result = await job.poll(ack_seq, wait)

        # Auto-cleanup after terminal state. Use pop to avoid KeyError if a
        # concurrent kill() already removed the job between our await and here.
        if result.state in ("completed", "killed"):
            if self._jobs.pop(pid, None) is not None:
                job.retire()
                self._retired_jobs.append(job)
                await job.cleanup()
                self._keep_final_reply(job, self._running_keys.pop(pid, None))

        return result

    def _final_reply(self, pid: int, ack_seq: int, job_key: str | None) -> PollResult:
        """The final reply of the ended job `job_key` again, if it is still kept."""
        self._drop_final_replies()
        kept = self._final_replies.get(job_key) if job_key is not None else None
        # The PID too: a new job may have been given an ended job's.
        if kept is None or kept[0].pid != pid:
            if job_key in self._restarted:
                raise ToolException(
                    f"No job found with pid {pid}: exec_remote job {job_key} {_RESTARTED}"
                )
            raise ToolException(f"No job found with pid {pid}")
        return kept[0].replay(ack_seq)

    def _keep_final_reply(self, job: Job, job_key: str | None) -> None:
        if job_key is None:
            job.forget_unacked()  # nothing can ask for it again
            return
        self._final_replies[job_key] = (job, time.monotonic())
        self._drop_final_replies()

    def _drop_final_replies(self) -> None:
        """Forget the final replies kept too long, then the oldest while too much is kept."""
        expired = time.monotonic() - _FINAL_REPLY_SECONDS
        kept = sum(job.unacked_chars for job, _ in self._final_replies.values())
        for key, (job, ended) in list(self._final_replies.items()):
            if ended > expired and kept <= _FINAL_REPLY_CHARS:
                break
            kept -= job.unacked_chars
            job.forget_unacked()
            del self._final_replies[key]

    async def kill(self, pid: int, ack_seq: int) -> KillResult:
        """Terminate a running job and return any remaining buffered output."""
        job = self._get_job(pid)
        seq, stdout, stderr = await job.kill(ack_seq)
        # Use pop to avoid KeyError if a concurrent poll() already removed the
        # job between our await and here.
        if self._jobs.pop(pid, None) is not None:
            job.retire()
            self._retired_jobs.append(job)
            await job.cleanup()
            self._running_keys.pop(pid, None)
            job.forget_unacked()  # a kill's reply is never asked for again
        return KillResult(seq=seq, stdout=stdout, stderr=stderr)

    async def write_stdin(self, pid: int, data: str, ack_seq: int) -> WriteStdinResult:
        """Write data to stdin of a running job and return buffered output."""
        job = self._get_job(pid)
        seq, stdout, stderr = await job.write_stdin(data, ack_seq)
        return WriteStdinResult(seq=seq, stdout=stdout, stderr=stderr)

    async def close_stdin(self, pid: int, ack_seq: int) -> CloseStdinResult:
        """Close stdin of a running job and return buffered output."""
        job = self._get_job(pid)
        seq, stdout, stderr = await job.close_stdin(ack_seq)
        return CloseStdinResult(seq=seq, stdout=stdout, stderr=stderr)

    async def shutdown(self) -> None:
        """Terminate every job owned by this server."""
        jobs = [*self._jobs.values(), *self._retired_jobs]
        self._jobs.clear()
        self._retired_jobs.clear()
        self._running_keys.clear()
        self._final_replies.clear()

        async def shutdown_job(job: Job) -> None:
            errors: list[Exception] = []
            try:
                await job.shutdown()
            except Exception as ex:
                errors.append(ex)
            try:
                await job.cleanup()
            except Exception as ex:
                errors.append(ex)
            if errors:
                raise RuntimeError("; ".join(str(error) for error in errors))

        results = await asyncio.gather(
            *(shutdown_job(job) for job in jobs), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def _get_job(self, pid: int) -> Job:
        """Get job by PID or raise error."""
        job = self._jobs.get(pid)
        if job is None:
            raise ToolException(f"No job found with pid {pid}")
        return job
