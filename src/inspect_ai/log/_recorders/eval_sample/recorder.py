"""Recorder for the `.eval.sample` directory-of-objects log format.

A thin shell over `SampleDirStore`: the live event stream is written by the
store (the same sink the `TaskLogger` drives for the realtime buffer), so the
recorder only manages the eval-level header (`start.json` / `header.json`) and
folds sample dirs back into `EvalSample`s on read via the store's one reader.
"""

from __future__ import annotations

import math
from typing import IO, TYPE_CHECKING, Any

from typing_extensions import override

from inspect_ai._util.async_zip import AsyncZipReader
from inspect_ai._util.constants import LOG_SCHEMA_VERSION
from inspect_ai._util.error import EvalError
from inspect_ai.log._edit import LogUpdate

from ..._log import (
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSampleReductions,
    EvalSampleSummary,
    EvalSpec,
    EvalStats,
    EvalStatus,
)
from ..eval import LogStart
from ..file import FileRecorder

# `store` pulls in buffer.types -> _display -> inspect_ai.log, so it can't be
# imported at module load (this module is registered while inspect_ai.log is
# initializing). Reference it under TYPE_CHECKING and import it inside methods —
# the same pattern EvalRecorder uses for buffer.history.
if TYPE_CHECKING:
    from .store import SampleDirStore


class EvalSampleRecorder(FileRecorder):
    def __init__(self, log_dir: str, fs_options: dict[str, Any] | None = None) -> None:
        super().__init__(log_dir, "", fs_options)
        self._stores: dict[str, SampleDirStore] = {}
        self._starts: dict[str, LogStart] = {}

    @classmethod
    @override
    def handles_location(cls, location: str) -> bool:
        from .store import is_eval_sample_log

        return is_eval_sample_log(location)

    @classmethod
    @override
    def handles_bytes(cls, first_bytes: bytes) -> bool:
        return False

    @override
    def default_log_buffer(self, sample_count: int, high_throughput: bool) -> int:
        if high_throughput:
            return max(10, sample_count // 20)
        return max(1, min(math.floor(sample_count / 3), 10))

    # ---- write ----------------------------------------------------------

    @override
    async def log_init(self, eval: EvalSpec, location: str | None = None) -> str:
        from .store import SampleDirStore

        path = location or self._log_file_path(eval)
        store = SampleDirStore(
            path,
            log_images=eval.config.log_images is not False,
            update_interval=eval.config.log_shared or 1,
        )
        self._stores[self._log_file_key(eval)] = store
        return path

    def store_for(self, eval: EvalSpec) -> SampleDirStore:
        """The live store for this eval (used as the TaskLogger's event sink)."""
        return self._stores[self._log_file_key(eval)]

    @override
    async def log_start(self, eval: EvalSpec, plan: EvalPlan) -> None:
        key = self._log_file_key(eval)
        store = self._stores[key]
        self._starts[key] = LogStart(version=LOG_SCHEMA_VERSION, eval=eval, plan=plan)
        # the run header (eval/plan/status) is written into each sample dir as
        # header.jsonl when the sample starts — set it on the store here
        store.run_header = EvalLog(
            version=LOG_SCHEMA_VERSION, eval=eval, plan=plan, status="started"
        )

    @override
    async def log_sample(self, eval: EvalSpec, sample: EvalSample) -> None:
        # events stream live into the store; here we append the full footer
        self._stores[self._log_file_key(eval)].write_footer(sample)

    @override
    async def flush(self, eval: EvalSpec) -> None:
        self._stores[self._log_file_key(eval)].flush()

    @override
    async def log_finish(
        self,
        eval: EvalSpec,
        status: EvalStatus,
        stats: EvalStats,
        results: EvalResults | None,
        reductions: list[EvalSampleReductions] | None,
        error: EvalError | None = None,
        header_only: bool = False,
        invalidated: bool = False,
        log_updates: list[LogUpdate] | None = None,
    ) -> EvalLog:
        key = self._log_file_key(eval)
        store = self._stores[key]
        start = self._starts.get(key)
        eval_spec = start.eval if start else eval
        plan = start.plan if start else EvalPlan()

        # stamp the final eval-level status into every sample dir's header.jsonl
        # (run-level config + status only — NOT cross-run aggregates, which are
        # recomputed at compaction to .eval)
        run = EvalLog(
            version=LOG_SCHEMA_VERSION,
            eval=eval_spec,
            plan=plan,
            status=status,
            error=error,
            invalidated=invalidated,
            log_updates=log_updates,
        )
        store.write_run_header(run)

        # the returned (in-memory) log carries the framework-computed aggregates
        # for the console summary, hooks, and the eval() return value
        log = run.model_copy(update={"results": results, "stats": stats})
        log.location = store._dir
        if not header_only:
            log.reductions = reductions
            log.samples = [
                store.read_eval_sample(s.id, s.epoch) for s in store.sample_summaries()
            ]

        del self._stores[key]
        self._starts.pop(key, None)
        return log

    # ---- read -----------------------------------------------------------

    @classmethod
    @override
    async def read_log(cls, location: str, header_only: bool = False) -> EvalLog:
        from .store import SampleDirStore

        store = SampleDirStore(location, create=False)
        # the run header (eval/plan/status) lives in each sample dir's
        # header.jsonl; results/stats/reductions are recomputed at compaction
        log = store.read_run_header() or EvalLog(
            version=LOG_SCHEMA_VERSION, eval=_empty_spec(), status="started"
        )
        log.location = location
        if header_only:
            return log
        log.samples = [
            store.read_eval_sample(s.id, s.epoch) for s in store.sample_summaries()
        ]
        return log

    @override
    @classmethod
    async def read_log_bytes(
        cls, log_bytes: IO[bytes], header_only: bool = False
    ) -> EvalLog:
        raise NotImplementedError(
            "The eval.sample format is a directory and cannot be read from bytes."
        )

    @override
    @classmethod
    async def read_log_sample(
        cls,
        location: str,
        id: str | int | None = None,
        epoch: int = 1,
        uuid: str | None = None,
        exclude_fields: set[str] | None = None,
        reader: AsyncZipReader | None = None,
    ) -> EvalSample:
        from .store import SampleDirStore

        store = SampleDirStore(location, create=False)
        if id is None:
            if uuid is None:
                raise ValueError("You must specify an 'id' or 'uuid' to read")
            match = next((s for s in store.sample_summaries() if s.uuid == uuid), None)
            if match is None:
                raise IndexError(f"Sample with uuid '{uuid}' not found in {location}")
            id, epoch = match.id, match.epoch
        return store.read_eval_sample(id, epoch)

    @classmethod
    @override
    async def read_log_sample_summaries(cls, location: str) -> list[EvalSampleSummary]:
        from .store import SampleDirStore

        return SampleDirStore(location, create=False).sample_summaries()

    @classmethod
    @override
    async def write_log(
        cls,
        location: str,
        log: EvalLog,
        if_match_etag: str | None = None,
        header_only: bool = False,
    ) -> None:
        raise NotImplementedError(
            "The eval.sample format is append-only; compact to .eval to edit."
        )


def _empty_spec() -> EvalSpec:
    return EvalSpec.model_construct()
