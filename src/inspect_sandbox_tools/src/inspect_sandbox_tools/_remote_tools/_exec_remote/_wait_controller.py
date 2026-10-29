"""Request-owned commands with bounded waits and terminal-response replay."""

import asyncio
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import NamedTuple

from inspect_sandbox_tools._util.common_types import ToolException

from ._job import Job
from .tool_types import KillResult, PollResult, StartWaitParams, WaitResult

_TERMINAL_TTL = 300.0
_MAX_ACKNOWLEDGED_ENTRIES = 256
_MAX_RETAINED_BYTES = 64 * 1024 * 1024
_EXPIRED_ID_TTL = 3600.0
_MAX_EXPIRED_IDS = 50_000  # about 13 MB
_LAUNCH_SHUTDOWN_GRACE = 1.0
_RESTARTED = "may have run before the sandbox tools server restarted"


@dataclass
class _Request:
    job: Job | None = None
    launch: asyncio.Task[None] | None = None
    stop: asyncio.Task[KillResult] | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Serialize output/ACK mutations; never hold this while waiting for new output.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancelled: bool = False
    error: Exception | None = None
    response: PollResult | None = None
    finished_at: float | None = None
    acknowledged: bool = False
    seen_at: float = 0.0  # last start or poll of this request


def _retained_bytes(entry: _Request) -> int:
    """Memory held by a request's stored reply and unacknowledged output chunks.

    Once the final reply is stored the job's chunks are released, so a finished
    request counts only that reply.
    """
    size = 0
    if entry.response is not None:
        size += sys.getsizeof(entry.response.stdout)
        size += sys.getsizeof(entry.response.stderr)
    if entry.job is not None:
        size += entry.job.held_output_bytes
    return size


class _Expired(NamedTuple):
    at: float
    outcome: int | str
    """The exit code of a command that completed, or what else happened to it."""


def _outcome(entry: _Request) -> int | str:
    response = entry.response
    if response is not None and response.exit_code is not None:
        return response.exit_code
    if response is not None and response.state == "killed":
        return "ran and was killed"
    if entry.error is not None and entry.job is None:
        return f"could not be started ({entry.error!r})"
    if entry.error is not None:
        return f"was stopped when its input could not be written ({entry.error!r})"
    if entry.job is not None and entry.job.exited:
        return "exited, but its final reply was dropped before it was sent"
    return "was cancelled before it started"


class WaitController:
    """Own launches independently of RPC cancellation and retain bounded replay.

    Request IDs must never be reused by clients. A repeated start with a known
    ID reports that request instead of launching again. Retention is described
    in `_prune`; a missing entry is an explicit error, never an implicit new launch.
    With `persist_launches`, a restarted server also knows the IDs the previous one
    launched, so a start repeated across a restart is reported lost, not run again.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _Request] = {}
        self._expired: OrderedDict[str, _Expired] = OrderedDict()
        self._retired_jobs: list[Job] = []
        self._launched_path: Path | None = None

    def persist_launches(self, path: Path) -> None:
        """Record each request ID in `path` before launching it, and load earlier ones.

        A client resends a start whose reply was lost (and the sandbox transport
        retries a dropped call), so a server restarted in between must know which
        IDs the previous one already launched.
        """
        try:
            launched = path.read_text().split()
        except FileNotFoundError:
            launched = []
        now = monotonic()
        for request_id in launched[-_MAX_EXPIRED_IDS:]:
            self._expired[request_id] = _Expired(now, _RESTARTED)
        self._launched_path = path

    def _record_launch(self, request_id: str) -> None:
        if self._launched_path is None:
            return
        try:
            with self._launched_path.open("a") as launched:
                launched.write(f"{request_id}\n")
        except OSError as ex:
            # Best effort: a full disk must not stop commands (such as cleanup) running.
            print(
                f"exec_remote: could not record request {request_id}: {ex}",
                file=sys.stderr,
            )

    async def start(
        self, params: StartWaitParams, can_switch_user: bool = False
    ) -> WaitResult:
        """Register a unique request before launching and wait briefly for completion."""
        self._acknowledge(params.acks)
        self._prune()
        if params.request_id in self._requests or params.request_id in self._expired:
            # A retried start (e.g. after a dropped connection) acts as a poll; a
            # cancelled or expired request raises instead of launching late.
            return await self.poll(params.request_id, 0, params.wait_seconds)
        if params.stdin_open:
            raise ToolException("Awaitable exec_remote requests cannot keep stdin open")
        entry = _Request(seen_at=monotonic())
        self._requests[params.request_id] = entry
        self._record_launch(params.request_id)
        entry.launch = asyncio.create_task(self._launch(entry, params, can_switch_user))
        await asyncio.shield(entry.launch)
        job = self._job(entry, params.request_id)
        await job.wait_for_update(params.wait_seconds)
        return await self._poll(entry, params.request_id, 0)

    async def _launch(
        self, entry: _Request, params: StartWaitParams, can_switch_user: bool
    ) -> None:
        def register(job: Job) -> None:
            entry.job = job
            entry.ready.set()

        try:
            await Job.create(
                params.command,
                input=params.input,
                stdin_open=params.stdin_open,
                env=params.env,
                cwd=params.cwd,
                user=params.user,
                can_switch_user=can_switch_user,
                on_create=register,
            )
        except Exception as ex:
            if not entry.cancelled:
                entry.error = ex
                entry.finished_at = monotonic()
        finally:
            entry.ready.set()

    async def poll(
        self, request_id: str, ack_seq: int, wait_seconds: float
    ) -> WaitResult:
        """Wait for output/exit, replaying an unacknowledged response immediately."""
        entry = self._get(request_id)
        if entry.launch is not None:
            await entry.ready.wait()
        job = self._job(entry, request_id)
        if entry.response is None or ack_seq >= entry.response.seq:
            await job.wait_for_update(wait_seconds)
        return await self._poll(entry, request_id, ack_seq)

    async def _poll(self, entry: _Request, request_id: str, ack_seq: int) -> WaitResult:
        async with entry.lock:
            if entry.finished_at is None and request_id not in self._requests:
                # dropped under the byte cap while this poll waited: its output
                # was released, so report it lost rather than complete empty
                self._get(request_id)
            entry.seen_at = monotonic()
            job = self._job(entry, request_id)
            if entry.response is not None and ack_seq < entry.response.seq:
                result = entry.response
            else:
                result = await job.poll(ack_seq, terminal_output=True)
                entry.response = result
            if result.state != "running" and entry.finished_at is None:
                entry.finished_at = monotonic()
                job.retire()
                await job.cleanup()
                job.release_output()  # the stored reply is the copy kept for replay
            response = WaitResult(pid=job.pid, result=result)
            self._prune()
            return response

    async def kill(self, request_id: str, ack_seq: int) -> KillResult:
        """Cancel a launch even when the start RPC has not registered it yet."""
        self._prune()
        entry = self._requests.get(request_id)
        if entry is None:
            if request_id not in self._expired:
                self._requests[request_id] = _Request(
                    cancelled=True, finished_at=monotonic()
                )
                self._prune()
            return KillResult(seq=0, stdout="", stderr="")
        entry.cancelled = True
        if entry.stop is None:
            entry.stop = asyncio.create_task(self._stop(entry, ack_seq))
            entry.stop.add_done_callback(lambda task: self._stop_finished(entry, task))
        result = await asyncio.shield(entry.stop)
        # A repeated kill with a newer ACK must not return output already delivered.
        if ack_seq >= result.seq:
            return KillResult(seq=result.seq, stdout="", stderr="")
        return result

    @staticmethod
    def _stop_finished(entry: _Request, task: asyncio.Task[KillResult]) -> None:
        # A cancelled RPC must not retain an unaccounted copy of its output.
        if entry.stop is task:
            entry.stop = None
        if not task.cancelled():
            task.exception()

    async def _stop(self, entry: _Request, ack_seq: int) -> KillResult:
        if entry.launch is not None:
            await entry.ready.wait()
        async with entry.lock:
            if entry.job is None:
                if entry.finished_at is None:
                    entry.finished_at = monotonic()
                self._prune()
                return KillResult(seq=0, stdout="", stderr="")
            if entry.response is not None and entry.response.state != "running":
                # The command already ended: keep its final reply for replay.
                response = entry.response
                return KillResult(
                    seq=response.seq, stdout=response.stdout, stderr=response.stderr
                )
            seq, stdout, stderr = await entry.job.kill(ack_seq, drain_terminal=True)
            entry.response = PollResult(
                state="killed", seq=seq, stdout=stdout, stderr=stderr
            )
            if entry.finished_at is None:
                entry.finished_at = monotonic()
            entry.job.retire()
            await entry.job.cleanup()
            entry.job.release_output()
            result = KillResult(seq=seq, stdout=stdout, stderr=stderr)
            self._prune()
            return result

    def _get(self, request_id: str) -> _Request:
        self._prune()
        entry = self._requests.get(request_id)
        if entry is None:
            if (expired := self._expired.get(request_id)) is not None:
                outcome = expired.outcome
                if isinstance(outcome, int):
                    outcome = f"ran and exited with code {outcome}"
                raise ToolException(
                    f"exec_remote request {request_id} result lost: the command "
                    f"{outcome}. Its output is no longer kept, and the command is "
                    "not run again."
                )
            raise ToolException(f"Unknown or expired exec_remote request {request_id}")
        return entry

    @staticmethod
    def _job(entry: _Request, request_id: str) -> Job:
        if entry.error is not None:
            # Raised as exec_remote_start raises it, so the host sees the same error.
            raise entry.error
        if entry.job is None:
            raise ToolException(
                f"exec_remote request {request_id} was cancelled before it started"
            )
        return entry.job

    def _acknowledge(self, request_ids: list[str]) -> None:
        """Drop output the host has received, keeping the duplicate-start marker."""
        for request_id in request_ids:
            entry = self._requests.get(request_id)
            if entry is None or entry.response is None or entry.finished_at is None:
                continue
            entry.acknowledged = True
            entry.response = entry.response.model_copy(
                update={"stdout": "", "stderr": ""}
            )
            if entry.job is not None:
                entry.job.release_output()

    def _prune(self) -> None:
        """Expire finished requests.

        Guarantee: a terminal reply the host has not acknowledged stays
        replayable for `_TERMINAL_TTL` seconds after it was first sent, unless
        unacknowledged replies together hold more than `_MAX_RETAINED_BYTES` of
        output, when the oldest are dropped first. A command that exited but
        whose final reply has not been sent yet (a host polling every long
        `poll_interval`) has no time limit, like a running command; it counts
        toward the byte cap with its last reply and unacknowledged output (not
        output still unread from its pipes, which is bounded per command as
        for a running one) and is dropped only after every sent reply, and
        never while a poll or kill holds its lock to make that reply (which
        would then report the end of the command without the released
        output). Hosts
        acknowledge terminal replies on a later start; acknowledged entries hold
        no output and only stop a late duplicate start, and the oldest beyond
        `_MAX_ACKNOWLEDGED_ENTRIES` are dropped early.
        A dropped request's ID is kept, with how the command ended but without
        output, for `_EXPIRED_ID_TTL` (at most `_MAX_EXPIRED_IDS` of them), so a
        late repeated start or replay of it fails, saying the result was lost
        and how the command ended, instead of launching it again.
        """
        now = monotonic()
        acknowledged: list[tuple[str, _Request]] = []
        unacknowledged: list[tuple[str, _Request]] = []
        # sent replies oldest first, then exited commands whose reply is unsent
        for key, entry in sorted(
            self._requests.items(),
            key=lambda item: (
                item[1].finished_at is None,
                item[1].finished_at or item[1].seen_at,
            ),
        ):
            if entry.finished_at is None:
                job = entry.job
                if job is not None and job.exited and not entry.lock.locked():
                    unacknowledged.append((key, entry))
            elif now - entry.finished_at >= _TERMINAL_TTL:
                self._evict(key, entry)
            elif entry.acknowledged:
                acknowledged.append((key, entry))
            else:
                unacknowledged.append((key, entry))
        excess = len(acknowledged) - _MAX_ACKNOWLEDGED_ENTRIES
        for key, entry in acknowledged[: max(excess, 0)]:
            self._evict(key, entry)
        retained = sum(_retained_bytes(entry) for _, entry in unacknowledged)
        for key, entry in unacknowledged:
            if retained <= _MAX_RETAINED_BYTES:
                break
            retained -= _retained_bytes(entry)
            self._evict(key, entry)
        while self._expired and (
            len(self._expired) > _MAX_EXPIRED_IDS
            or now - next(iter(self._expired.values())).at >= _EXPIRED_ID_TTL
        ):
            self._expired.popitem(last=False)

    def _evict(self, key: str, entry: _Request) -> None:
        self._requests.pop(key, None)
        self._expired[key] = _Expired(monotonic(), _outcome(entry))
        if entry.job is not None:
            entry.job.retire()
            entry.job.release_output()
            self._retired_jobs.append(entry.job)

    async def shutdown(self) -> None:
        """Terminate pending, active, and retired commands owned by this controller."""
        entries = list(self._requests.values())
        for entry in entries:
            entry.cancelled = True
        # Starting cancellation before awaiting launch also unblocks stdin.drain().
        results = await asyncio.gather(
            *(self._stop(entry, 0) for entry in entries), return_exceptions=True
        )
        jobs = set(entry.job for entry in entries if entry.job is not None) | set(
            self._retired_jobs
        )
        # Before awaiting launches: a launch can be blocked writing input to a
        # descendant that holds stdin open after the leader exited.
        shutdown_results = await asyncio.gather(
            *(job.shutdown() for job in jobs), return_exceptions=True
        )
        launches = [entry.launch for entry in entries if entry.launch is not None]
        if launches:
            # A process that left the job's process group can still hold stdin.
            _, blocked = await asyncio.wait(launches, timeout=_LAUNCH_SHUTDOWN_GRACE)
            for launch in blocked:
                launch.cancel()
            await asyncio.gather(*launches, return_exceptions=True)
        self._requests.clear()
        self._retired_jobs.clear()
        errors = [
            result
            for result in [*results, *shutdown_results]
            if isinstance(result, Exception)
        ]
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))
