"""Durable per-sample directory store for the `.eval.sample` format.

A single `SampleDirStore` owns one eval folder — a plain directory holding one
`<id>_<epoch>_<uuid>.eval.sample/` subdirectory per executed sample. Each sample
dir is a self-contained append-only log: a `header.jsonl` (the run-level
eval/plan/status, also the format marker), a `run_start` sample header, one
immutable object per event under `events/`, content-addressed `attachments/`,
message/call dedup pools under `message_pool/`+`call_pool/` (so repeated
ModelEvent inputs are stored once), and a slim `run_end` footer (set-at-end
fields only) whose presence marks completion.

The same object is the live write sink during an eval (driven by the same
`TaskLogger` calls that feed the SQLite buffer today) and the `SampleBuffer`
read source the viewer and recorder fold back into `EvalSample`s — so one path
serves running and completed samples alike. Listing the directory is the
source of truth, which makes a crashed run readable with no recovery step.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from logging import getLogger
from typing import Any, Iterable, Iterator, Literal

from typing_extensions import override

from inspect_ai._display.core.display import TaskDisplayMetric
from inspect_ai._util.file import (
    basename,
    clean_filename_component,
    filesystem,
    local_path,
    open_file,
)
from inspect_ai._util.json import to_json_safe
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._pool import (
    condense_model_event_calls,
    condense_model_event_inputs,
)

from ..._condense import condense_event
from ..._log import EvalLog, EvalSample, EvalSampleSummary
from ..._recover._reconstruct import reconstruct_eval_sample
from ..buffer.history import SampleHistory
from ..buffer.types import (
    AttachmentData,
    CallPoolData,
    EventData,
    MessagePoolData,
    SampleData,
    Samples,
    WritableSampleBuffer,
)
from ..types import SampleEvent

logger = getLogger(__name__)

SAMPLE_DIR_SUFFIX = ".eval.sample"
# the run-level header (eval/plan/status) lives inside EACH sample dir so a
# sample is fully self-contained; it also serves as the per-dir format marker
HEADER_JSON = "header.jsonl"
RUN_START = "run_start.jsonl"
RUN_END = "run_end.jsonl"
EVENTS_DIR = "events"
ATTACHMENTS_DIR = "attachments"
# message/call dedup pools (one object per entry, like events) — repeated
# ModelEvent inputs/calls are pooled into these and referenced by index
MESSAGE_POOL_DIR = "message_pool"
CALL_POOL_DIR = "call_pool"
CHECKPOINTS_DIR = "checkpoints"
AGENT_STATE = "agent_state.jsonl"

# the run header is the EvalLog minus cross-run aggregates and the sample list
_RUN_HEADER_EXCLUDE = {"results", "stats", "reductions", "samples"}

# the footer holds only the irreducible set-at-end fields (not the transcript,
# which is folded from the events, nor input/target, which are in the header /
# sample_init event)
_FOOTER_FIELDS = {
    "output",
    "scores",
    "store",
    "model_usage",
    "role_usage",
    "completed_at",
    "total_time",
    "working_time",
    "error",
    "limit",
    "error_retries",
}


def is_eval_sample_log(location: str) -> bool:
    """Whether `location` is an eval.sample log (an eval folder or a lone sample dir)."""
    loc = location.rstrip("/\\")
    if loc.endswith(SAMPLE_DIR_SUFFIX):
        return True
    try:
        fs = filesystem(location)
        # a lone sample dir carries header.jsonl; an eval folder holds .eval.sample dirs
        if fs.exists(f"{loc}{fs.sep}{HEADER_JSON}"):
            return True
        return any(basename(e.name).endswith(SAMPLE_DIR_SUFFIX) for e in fs.ls(loc))
    except Exception:
        return False


def _jsonl(x: object) -> str:
    """Compact, single-line, NaN/surrogate-safe JSON for one cat-able record."""
    return to_json_safe(x, indent=None).decode("utf-8")


def _event_type(event: object) -> str:
    return str(getattr(event, "event", "event"))


def _event_timestamp(event: object) -> int:
    ts = getattr(event, "timestamp", None)
    if isinstance(ts, datetime):
        return int(ts.timestamp())
    return 0


def _overlay_sample_init(sample: EvalSample) -> EvalSample:
    """Recover full input/target/etc from the sample_init event (the summary is thinned)."""
    from inspect_ai.event._sample_init import SampleInitEvent

    init = next((e for e in sample.events if isinstance(e, SampleInitEvent)), None)
    if init is None:
        return sample
    s = init.sample
    update: dict[str, object] = {"input": s.input}
    if s.target is not None:
        update["target"] = s.target
    if s.choices is not None:
        update["choices"] = s.choices
    if s.metadata:
        update["metadata"] = s.metadata
    if s.sandbox is not None:
        update["sandbox"] = s.sandbox
    return sample.model_copy(update=update)


class SampleDirStore(WritableSampleBuffer):
    def __init__(
        self,
        location: str,
        *,
        create: bool = True,
        log_images: bool = True,
        update_interval: int = 1,
    ) -> None:
        self._dir = location.rstrip("/\\")
        self._fs = filesystem(location)
        self._log_images = log_images
        self.update_interval = update_interval

        # writer-side state (empty for read-only opens)
        self._summaries: dict[tuple[str, int], EvalSampleSummary] = {}
        self._sample_dirs: dict[tuple[str, int], str] = {}
        self._event_index: dict[tuple[str, int], int] = {}
        self._attachments_written: dict[tuple[str, int], set[str]] = {}
        # per-sample message/call pool dedup indices (hash -> pool position),
        # carried across events like the SQLite buffer does
        self._msg_index: dict[tuple[str, int], dict[str, int]] = {}
        self._call_index: dict[tuple[str, int], dict[str, int]] = {}
        # events can arrive before start_sample (the SampleInitEvent precedes it);
        # hold the handful that do until the dir + uuid are known
        self._pending_events: dict[tuple[str, int], list[SampleEvent]] = {}
        self._metrics: list[TaskDisplayMetric] = []
        # run-level header (eval/plan/status) the recorder sets; written into
        # each sample dir as header.jsonl so every sample is self-contained
        self.run_header: EvalLog | None = None

        if create:
            self._fs.mkdir(self._dir, exist_ok=True)

    # ---- paths -----------------------------------------------------------

    def _join(self, *parts: str) -> str:
        return self._fs.sep.join([self._dir, *parts])

    def _sample_dir_name(self, summary: EvalSampleSummary) -> str:
        sid = clean_filename_component(str(summary.id))
        uuid = summary.uuid or "nouuid"
        return f"{sid}_{summary.epoch}_{uuid}{SAMPLE_DIR_SUFFIX}"

    # ---- write path (live sink) -----------------------------------------

    def start_sample(self, sample: EvalSampleSummary) -> None:
        key = (str(sample.id), sample.epoch)
        sample_dir = self._join(self._sample_dir_name(sample))
        self._fs.mkdir(sample_dir, exist_ok=True)
        for sub in (EVENTS_DIR, ATTACHMENTS_DIR, MESSAGE_POOL_DIR, CALL_POOL_DIR):
            self._fs.mkdir(f"{sample_dir}{self._fs.sep}{sub}", exist_ok=True)

        self._sample_dirs[key] = sample_dir
        self._event_index.setdefault(key, 0)
        self._attachments_written.setdefault(key, set())
        self._msg_index.setdefault(key, {})
        self._call_index.setdefault(key, {})
        self._summaries[key] = sample

        # each sample dir is self-contained: the run header (also the format
        # marker) + the sample's own start header
        self._write_header(sample_dir)
        self._write_text(f"{sample_dir}{self._fs.sep}{RUN_START}", _jsonl(sample))

        # flush any events that arrived before this start
        for pending in self._pending_events.pop(key, []):
            self._write_event(key, sample_dir, pending)

    def log_events(self, events: list[SampleEvent]) -> None:
        for sample_event in events:
            key = (str(sample_event.id), sample_event.epoch)
            sample_dir = self._sample_dirs.get(key)
            if sample_dir is None:
                self._pending_events.setdefault(key, []).append(sample_event)
                continue
            self._write_event(key, sample_dir, sample_event)

    def _write_event(
        self, key: tuple[str, int], sample_dir: str, sample_event: SampleEvent
    ) -> None:
        # extract large content into content-addressed attachment objects
        attachments: dict[str, str] = {}
        event = condense_event(sample_event.event, attachments, self._log_images)

        written = self._attachments_written.setdefault(key, set())
        for hash, content in attachments.items():
            if hash not in written:
                self._write_text(
                    f"{sample_dir}{self._fs.sep}{ATTACHMENTS_DIR}{self._fs.sep}{hash}",
                    content,
                )
                written.add(hash)

        # dedup repeated ModelEvent inputs/calls into the pools (same condense
        # functions the SQLite buffer uses) so storage is O(N), not O(N^2)
        if isinstance(event, ModelEvent):
            event = self._pool_model_event(key, sample_dir, event)

        index = self._event_index.get(key, 0)
        self._event_index[key] = index + 1
        name = f"{index:06d}_{_event_timestamp(event)}_{_event_type(event)}.jsonl"
        self._write_text(
            f"{sample_dir}{self._fs.sep}{EVENTS_DIR}{self._fs.sep}{name}",
            _jsonl(event),
        )

    def _pool_model_event(
        self, key: tuple[str, int], sample_dir: str, event: ModelEvent
    ) -> ModelEvent:
        # message pool: replace input with refs, append newly-seen messages
        msg_index = self._msg_index.setdefault(key, {})
        start = len(msg_index)
        [pooled], new_msg_index, new_msgs = condense_model_event_inputs(
            [event], start, msg_index
        )
        self._msg_index[key] = new_msg_index
        for i, (_hash, msg) in enumerate(new_msgs):
            self._write_text(
                f"{sample_dir}{self._fs.sep}{MESSAGE_POOL_DIR}{self._fs.sep}{start + i:06d}.jsonl",
                _jsonl(msg),
            )

        # call pool: same for ModelEvent.call
        call_index = self._call_index.setdefault(key, {})
        cstart = len(call_index)
        [pooled], new_call_index, new_calls = condense_model_event_calls(
            [pooled], cstart, call_index
        )
        self._call_index[key] = new_call_index
        for i, (_hash, call) in enumerate(new_calls):
            self._write_text(
                f"{sample_dir}{self._fs.sep}{CALL_POOL_DIR}{self._fs.sep}{cstart + i:06d}.jsonl",
                _jsonl(call),
            )
        return pooled  # type: ignore[return-value]

    def write_footer(self, sample: EvalSample) -> None:
        """Write run_end.jsonl: only the set-at-end fields (no transcript)."""
        key = (str(sample.id), sample.epoch)
        sample_dir = self._sample_dirs.get(key)
        if sample_dir is None:
            return
        footer = sample.model_dump(include=_FOOTER_FIELDS, exclude_none=True)
        self._write_text(f"{sample_dir}{self._fs.sep}{RUN_END}", _jsonl(footer))

    # ---- run header (eval/plan/status) per sample dir --------------------

    def _write_header(self, sample_dir: str) -> None:
        if self.run_header is not None:
            self._write_text(
                f"{sample_dir}{self._fs.sep}{HEADER_JSON}",
                self.run_header.model_dump_json(exclude=_RUN_HEADER_EXCLUDE),
            )

    def write_run_header(self, header: EvalLog) -> None:
        """Set the run header and (re)write it into every sample dir.

        Called at finish to stamp the final eval-level status into each
        self-contained sample dir.
        """
        self.run_header = header
        for sample_dir in self._scan_sample_dirs().values():
            self._write_header(sample_dir)

    def read_run_header(self) -> EvalLog | None:
        """Read the run header from any sample dir (they're identical)."""
        for sample_dir in self._scan_sample_dirs().values():
            text = self._read_text(f"{sample_dir}{self._fs.sep}{HEADER_JSON}")
            if text is not None:
                return EvalLog.model_validate_json(text)
        return None

    def complete_sample(self, summary: EvalSampleSummary) -> None:
        # completion is marked by run_end.jsonl (written via the recorder's
        # footer); the directory listing is the source of truth, so nothing
        # cross-sample is written here
        self._summaries[(str(summary.id), summary.epoch)] = summary

    def update_metrics(self, metrics: list[TaskDisplayMetric]) -> None:
        self._metrics = metrics

    def remove_samples(self, samples: list[tuple[str | int, int]]) -> None:
        # no-op: the directory IS the durable record, nothing to free
        return

    def flush(self) -> None:
        # events/attachments are written as produced; nothing is buffered
        return

    def _write_text(self, path: str, text: str) -> None:
        with open_file(path, "w") as f:
            f.write(text)

    # ---- read path (SampleBuffer) ---------------------------------------

    @classmethod
    @override
    def running_tasks(cls, log_dir: str) -> list[str] | None:
        return None

    def _read_text(self, path: str) -> str | None:
        try:
            with open_file(path, "r") as f:
                return str(f.read())
        except FileNotFoundError:
            return None

    def _scan_sample_dirs(self) -> dict[tuple[str, int], str]:
        """Resolve (id, epoch) -> sample dir by listing + reading run_start."""
        result: dict[tuple[str, int], str] = {}
        # a store may point directly at a single sample dir
        if self._dir.endswith(SAMPLE_DIR_SUFFIX):
            summary = self._read_summary(self._dir)
            if summary is not None:
                result[(str(summary.id), summary.epoch)] = self._dir
            return result
        try:
            entries = self._fs.ls(self._dir)
        except FileNotFoundError:
            return result
        for info in entries:
            name = basename(info.name)
            if not name.endswith(SAMPLE_DIR_SUFFIX):
                continue
            summary = self._read_summary(info.name)
            if summary is not None:
                result[(str(summary.id), summary.epoch)] = info.name
        return result

    def _read_summary(self, sample_dir: str) -> EvalSampleSummary | None:
        start_text = self._read_text(f"{sample_dir}{self._fs.sep}{RUN_START}")
        if start_text is None:
            return None
        base = json.loads(start_text)
        # overlay the summary-relevant terminal fields from the slim footer
        end_text = self._read_text(f"{sample_dir}{self._fs.sep}{RUN_END}")
        if end_text is not None:
            end = json.loads(end_text)
            base["completed"] = True
            for f in (
                "scores",
                "model_usage",
                "role_usage",
                "completed_at",
                "total_time",
                "working_time",
            ):
                if end.get(f) is not None:
                    base[f] = end[f]
            if end.get("error"):
                err = end["error"]
                base["error"] = (
                    err.get("message") if isinstance(err, dict) else str(err)
                )
        return EvalSampleSummary.model_validate(base)

    def _is_complete(self, sample_dir: str) -> bool:
        return self._fs.exists(f"{sample_dir}{self._fs.sep}{RUN_END}")

    def _read_events(
        self, sample_dir: str, after_event_id: int | None
    ) -> list[EventData]:
        events_dir = f"{sample_dir}{self._fs.sep}{EVENTS_DIR}"
        try:
            entries = self._fs.ls(events_dir)
        except FileNotFoundError:
            return []
        names = sorted(basename(info.name) for info in entries)
        result: list[EventData] = []
        sample_id = basename(sample_dir)
        for name in names:
            try:
                index = int(name.split("_", 1)[0])
            except ValueError:
                continue
            if after_event_id is not None and index <= after_event_id:
                continue
            text = self._read_text(f"{events_dir}{self._fs.sep}{name}")
            if text is None:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                # tolerate a torn trailing object from a crash mid-write
                continue
            result.append(
                EventData(
                    id=index,
                    event_id=str(event.get("uuid") or ""),
                    sample_id=sample_id,
                    epoch=0,
                    event=event,
                )
            )
        return result

    def _read_attachments(
        self, sample_dir: str, after_attachment_id: int | None
    ) -> list[AttachmentData]:
        att_dir = f"{sample_dir}{self._fs.sep}{ATTACHMENTS_DIR}"
        try:
            entries = self._fs.ls(att_dir)
        except FileNotFoundError:
            return []
        result: list[AttachmentData] = []
        sample_id = basename(sample_dir)
        for idx, info in enumerate(sorted(entries, key=lambda i: i.name), start=1):
            if after_attachment_id is not None and idx <= after_attachment_id:
                continue
            hash = basename(info.name)
            content = self._read_text(info.name)
            if content is None:
                continue
            result.append(
                AttachmentData(
                    id=idx, sample_id=sample_id, epoch=0, hash=hash, content=content
                )
            )
        return result

    @override
    def get_samples(
        self, etag: str | None = None
    ) -> Samples | Literal["NotModified"] | None:
        if not self._fs.exists(self._dir):
            return None
        summaries: list[EvalSampleSummary] = []
        completed = 0
        for sample_dir in self._scan_sample_dirs().values():
            summary = self._read_summary(sample_dir)
            if summary is None:
                continue
            summary.completed = self._is_complete(sample_dir)
            completed += int(summary.completed)
            summaries.append(summary)
        # etag changes when the sample list or any completion flips — the
        # viewer polls running samples' events separately by cursor
        version = f"{len(summaries)}:{completed}"
        if etag is not None and etag == version:
            return "NotModified"
        return Samples(
            samples=summaries,
            metrics=self._metrics,
            refresh=self.update_interval,
            etag=version,
        )

    def _resolve_sample_dir(self, id: str | int, epoch: int) -> str | None:
        key = (str(id), epoch)
        if key in self._sample_dirs:
            return self._sample_dirs[key]
        return self._scan_sample_dirs().get(key)

    @override
    def get_sample_data(
        self,
        id: str | int,
        epoch: int,
        after_event_id: int | None = None,
        after_attachment_id: int | None = None,
        after_message_pool_id: int | None = None,
        after_call_pool_id: int | None = None,
    ) -> SampleData | None:
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return None
        return SampleData(
            events=self._read_events(sample_dir, after_event_id),
            attachments=self._read_attachments(sample_dir, after_attachment_id),
            message_pool=[
                MessagePoolData(
                    id=idx,
                    sample_id=basename(sample_dir),
                    epoch=0,
                    msg_id="",
                    data=text,
                )
                for idx, text in self._read_pool(
                    sample_dir, MESSAGE_POOL_DIR, after_message_pool_id
                )
            ],
            call_pool=[
                CallPoolData(
                    id=idx, sample_id=basename(sample_dir), epoch=0, hash="", data=text
                )
                for idx, text in self._read_pool(
                    sample_dir, CALL_POOL_DIR, after_call_pool_id
                )
            ],
        )

    def _read_pool(
        self, sample_dir: str, pool_dir: str, after_id: int | None
    ) -> list[tuple[int, str]]:
        """Read pooled entries as (index, json-text), ordered by index."""
        path = f"{sample_dir}{self._fs.sep}{pool_dir}"
        try:
            entries = self._fs.ls(path)
        except FileNotFoundError:
            return []
        result: list[tuple[int, str]] = []
        for name in sorted(basename(e.name) for e in entries):
            try:
                idx = int(name.split(".", 1)[0])
            except ValueError:
                continue
            if after_id is not None and idx <= after_id:
                continue
            text = self._read_text(f"{path}{self._fs.sep}{name}")
            if text is not None:
                result.append((idx, text))
        return result

    @override
    def sample_event_count(self, id: str | int, epoch: int) -> int:
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return 0
        return len(self._read_events(sample_dir, None))

    def _history(
        self, id: str | int, epoch: int, events: Iterable[EventData]
    ) -> SampleHistory:
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return SampleHistory(list(events), [], [], {})
        attachments = {
            a.hash: a.content for a in self._read_attachments(sample_dir, None)
        }
        msg_pool = [
            json.loads(t)
            for _i, t in self._read_pool(sample_dir, MESSAGE_POOL_DIR, None)
        ]
        call_pool = [
            json.loads(t) for _i, t in self._read_pool(sample_dir, CALL_POOL_DIR, None)
        ]
        return SampleHistory(list(events), msg_pool, call_pool, attachments)

    @override
    @contextmanager
    def open_sample_history_tail(
        self, id: str | int, epoch: int, n: int
    ) -> Iterator[SampleHistory]:
        events = self.get_sample_data(id, epoch)
        rows = events.events[-n:] if (events and n > 0) else []
        yield self._history(id, epoch, rows)

    @override
    @contextmanager
    def open_sample_history_from(
        self, id: str | int, epoch: int, start: int
    ) -> Iterator[SampleHistory]:
        yield self._history(id, epoch, self._read_events_from(id, epoch, start))

    @override
    @contextmanager
    def open_sample_history(self, id: str | int, epoch: int) -> Iterator[SampleHistory]:
        data = self.get_sample_data(id, epoch)
        yield self._history(id, epoch, data.events if data else [])

    def _read_events_from(
        self, id: str | int, epoch: int, start: int
    ) -> list[EventData]:
        data = self.get_sample_data(id, epoch)
        if data is None:
            return []
        return [e for e in data.events if e.id >= start]

    @override
    def cleanup(self) -> None:
        # no-op: the directory IS the durable record
        return

    # ---- folding (one reader for running + completed) --------------------

    def read_eval_sample(self, id: str | int, epoch: int) -> EvalSample:
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            raise IndexError(f"Sample {id} epoch {epoch} not found in {self._dir}")
        summary = self._read_summary(sample_dir)
        if summary is None:
            raise IndexError(f"Sample {id} epoch {epoch} has no header in {self._dir}")
        data = self.get_sample_data(id, epoch) or SampleData(events=[], attachments=[])
        completed = self._is_complete(sample_dir)

        # fold the event stream — same machinery for running and completed
        sample = reconstruct_eval_sample(summary, data, cancelled=not completed)
        # the run_start summary is thinned; recover full input/target/etc from
        # the sample_init event so completed reads + compaction stay faithful
        sample = _overlay_sample_init(sample)
        if not completed:
            return sample

        # overlay the slim footer's set-at-end fields (output/scores/store/…)
        footer_text = self._read_text(f"{sample_dir}{self._fs.sep}{RUN_END}")
        if footer_text:
            base = sample.model_dump(mode="json", exclude={"events", "attachments"})
            base.update(json.loads(footer_text))
            sample = EvalSample.model_validate(base).model_copy(
                update={"events": sample.events, "attachments": sample.attachments}
            )
        return sample

    # ---- checkpoint resume state (the one host piece not in the stream) --

    def checkpoints_root(self, id: str | int, epoch: int) -> str | None:
        """The sample's `checkpoints/` dir, or ``None`` if the sample is unknown.

        Checkpointing nests its per-sample state *inside* the sample dir for
        this format — `checkpoints/<NNNN>/{checkpoint.json,agent_state.jsonl}`
        plus a shared `checkpoints/restic/` — so a sample directory is the
        single self-contained unit for resume.
        """
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return None
        # restic operates on local paths, so strip any `file://` the
        # filesystem listing added (s3:// and bare paths pass through)
        return f"{local_path(sample_dir)}{self._fs.sep}{CHECKPOINTS_DIR}"

    def _checkpoint_dir(self, sample_dir: str, checkpoint_id: int) -> str:
        return f"{sample_dir}{self._fs.sep}{CHECKPOINTS_DIR}{self._fs.sep}{checkpoint_id:04d}"

    def write_agent_state(
        self,
        id: str | int,
        epoch: int,
        checkpoint_id: int,
        agent_state: dict[str, Any],
    ) -> None:
        """Persist the cp.track() bag — the only host state not in the event stream."""
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return
        ckpt_dir = self._checkpoint_dir(sample_dir, checkpoint_id)
        self._fs.mkdir(ckpt_dir, exist_ok=True)
        self._write_text(f"{ckpt_dir}{self._fs.sep}{AGENT_STATE}", _jsonl(agent_state))

    def read_agent_state(
        self, id: str | int, epoch: int, checkpoint_id: int | None = None
    ) -> dict[str, Any] | None:
        sample_dir = self._resolve_sample_dir(id, epoch)
        if sample_dir is None:
            return None
        if checkpoint_id is None:
            checkpoint_id = self._latest_checkpoint_id(sample_dir)
            if checkpoint_id is None:
                return None
        text = self._read_text(
            f"{self._checkpoint_dir(sample_dir, checkpoint_id)}{self._fs.sep}{AGENT_STATE}"
        )
        return None if text is None else json.loads(text)

    def _latest_checkpoint_id(self, sample_dir: str) -> int | None:
        try:
            entries = self._fs.ls(f"{sample_dir}{self._fs.sep}{CHECKPOINTS_DIR}")
        except FileNotFoundError:
            return None
        ids = [int(basename(e.name)) for e in entries if basename(e.name).isdigit()]
        return max(ids) if ids else None

    def sample_summaries(self) -> list[EvalSampleSummary]:
        result: list[EvalSampleSummary] = []
        for sample_dir in self._scan_sample_dirs().values():
            summary = self._read_summary(sample_dir)
            if summary is None:
                continue
            summary.completed = self._is_complete(sample_dir)
            result.append(summary)
        return result
