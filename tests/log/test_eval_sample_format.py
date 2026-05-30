"""Tests for the opt-in `.eval.sample` directory-of-objects log format.

The format writes one append-only `<id>_<epoch>_<uuid>.eval.sample/` directory
per sample, fully self-contained: a `header.jsonl` (run config + status, also
the format marker), a `run_start` sample header, one cat-able object per event,
content-addressed attachments, and a slim `run_end` footer (set-at-end fields
only) marking completion. The eval folder is just these sample dirs. Cross-run
aggregates are recomputed at compaction, not persisted live. One reader folds
both running and completed samples.
"""

import glob
import json
import os

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.event._info import InfoEvent
from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_sample
from inspect_ai.log._log import EvalSampleSummary
from inspect_ai.log._recorders.buffer.buffer import sample_buffer
from inspect_ai.log._recorders.create import (
    recorder_type_for_format,
    recorder_type_for_location,
)
from inspect_ai.log._recorders.eval_sample import EvalSampleRecorder
from inspect_ai.log._recorders.eval_sample.store import SampleDirStore
from inspect_ai.log._recorders.types import SampleEvent
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


def _run(tmp_path, n: int = 2):
    task = Task(
        dataset=[Sample(input=f"q{i}", target="hello") for i in range(n)],
        solver=generate(),
        scorer=includes(),
    )
    return eval(
        task, model="mockllm/model", log_dir=str(tmp_path), log_format="eval.sample"
    )[0]


def test_round_trip_completed(tmp_path):
    log = _run(tmp_path, n=2)
    assert log.status == "success"
    # the log location is a directory of `.eval.sample` sub-dirs
    assert os.path.isdir(log.location)
    sample_dirs = glob.glob(os.path.join(log.location, "*.eval.sample"))
    assert len(sample_dirs) == 2

    re = read_eval_log(log.location)
    assert re.status == "success"
    assert re.samples is not None and len(re.samples) == 2
    for s in re.samples:
        assert len(s.messages) >= 1
        assert s.scores is not None and "includes" in s.scores
        assert len(s.events) > 0


def test_cat_ability(tmp_path):
    log = _run(tmp_path, n=1)
    sample_dir = glob.glob(os.path.join(log.location, "*.eval.sample"))[0]

    # every event object is a single line of valid JSON carrying its type
    event_files = sorted(glob.glob(os.path.join(sample_dir, "events", "*.jsonl")))
    assert len(event_files) > 0
    for ef in event_files:
        text = open(ef).read()
        assert text.count("\n") <= 1  # single record, cat-able
        assert "event" in json.loads(text)
    # the type is greppable from the filename
    assert any("_sample_init.jsonl" in os.path.basename(f) for f in event_files)


def test_sample_centric_layout(tmp_path):
    """The eval folder is ONLY sample dirs; each is self-contained with header.jsonl."""
    log = _run(tmp_path, n=2)
    entries = set(os.listdir(log.location))

    # the eval folder holds nothing but the sample dirs — no run-level files
    sample_dirs = {e for e in entries if e.endswith(".eval.sample")}
    assert len(sample_dirs) == 2
    assert entries == sample_dirs, (
        f"unexpected eval-folder entries: {entries - sample_dirs}"
    )

    for d in sample_dirs:
        files = set(os.listdir(os.path.join(log.location, d)))
        # the run header lives inside the sample dir (also the format marker)
        assert "header.jsonl" in files
        assert "run_start.jsonl" in files
        assert "run_end.jsonl" in files
        assert ".eval.sample.jsonl" not in files  # header.jsonl is the marker now

        # run_end is slim: set-at-end fields only, no transcript / start fields
        footer = json.load(open(os.path.join(log.location, d, "run_end.jsonl")))
        assert "scores" in footer  # a terminal field is present
        for absent in ("messages", "events", "attachments", "input", "target"):
            assert absent not in footer, f"run_end should not carry {absent}"


def test_raw_read_has_no_aggregates(tmp_path):
    """A raw (uncompacted) eval.sample folder carries no cross-run aggregates."""
    log = _run(tmp_path, n=2)
    re = read_eval_log(log.location)
    assert re.status == "success"  # status IS persisted (header.jsonl)
    assert re.samples is not None and len(re.samples) == 2
    assert re.results is None and re.reductions is None  # aggregates not persisted
    # but per-sample scores ARE present (they live in the sample footers)
    assert all(s.scores for s in re.samples)


def test_crash_mid_sample_viewable_without_recover(tmp_path):
    """A sample with no run_end footer (crashed) is still readable — no recover."""
    log = _run(tmp_path, n=1)
    sample_dir = glob.glob(os.path.join(log.location, "*.eval.sample"))[0]

    # simulate a crash before the footer was written
    os.remove(os.path.join(sample_dir, "run_end.jsonl"))

    re = read_eval_log(log.location)
    assert re.samples is not None and len(re.samples) == 1
    crashed = re.samples[0]
    # in-progress samples surface a synthesized cancellation, transcript intact
    assert crashed.error is not None
    assert "Cancelled" in crashed.error.message
    assert len(crashed.events) > 0


def test_read_single_sample(tmp_path):
    log = _run(tmp_path, n=2)
    summaries = read_eval_log(log.location).samples
    assert summaries is not None
    first = summaries[0]
    s = read_eval_log_sample(log.location, id=first.id, epoch=first.epoch)
    assert s.id == first.id
    # uuid-based lookup also resolves by scanning the sample dirs
    by_uuid = read_eval_log_sample(log.location, uuid=first.uuid)
    assert by_uuid.id == first.id


def test_format_routing(tmp_path):
    assert recorder_type_for_format("eval.sample") is EvalSampleRecorder
    # the eval folder is detected via its .eval.sample dirs; defaults untouched
    log = _run(tmp_path, n=1)
    assert recorder_type_for_location(log.location) is EvalSampleRecorder
    assert recorder_type_for_format("eval") is not EvalSampleRecorder
    # a lone sample dir is recognized by suffix
    sample_dir = glob.glob(os.path.join(log.location, "*.eval.sample"))[0]
    assert EvalSampleRecorder.handles_location(sample_dir)
    assert not EvalSampleRecorder.handles_location("/tmp/whatever.eval")


def test_auto_format_detection_reads_folder(tmp_path):
    log = _run(tmp_path, n=1)
    re = read_eval_log(log.location, format="auto")
    assert re.samples is not None and len(re.samples) == 1


def test_list_eval_logs_discovers_folder(tmp_path):
    # a .eval.sample folder and a normal .eval file coexist and both list
    sample_log = _run(tmp_path, n=1)
    eval_log = eval(
        Task(
            dataset=[Sample(input="hi", target="hello")],
            solver=generate(),
            scorer=includes(),
        ),
        model="mockllm/model",
        log_dir=str(tmp_path),
        log_format="eval",
    )[0]

    listed = {
        os.path.basename(li.name.rstrip("/")): li
        for li in list_eval_logs(str(tmp_path))
    }
    sample_name = os.path.basename(sample_log.location.rstrip("/"))
    assert sample_name in listed
    assert listed[sample_name].type == "dir"
    assert any(li.name.endswith(".eval") for li in listed.values())
    assert eval_log.location  # sanity


def test_memory_bounded_events_stream_to_disk(tmp_path):
    """Events are written as produced and never retained — memory stays flat.

    The store keeps only O(1) per-sample bookkeeping (an index counter and the
    set of attachment hashes), not the transcript, so a very long sample does
    not grow memory with its event count.
    """
    store = SampleDirStore(str(tmp_path / "run"))
    summary = EvalSampleSummary(id="1", epoch=1, input="x", target="y", uuid="u1")
    store.start_sample(summary)

    n = 500
    for i in range(n):
        store.log_events(
            [SampleEvent(id="1", epoch=1, event=InfoEvent(source="t", data=f"e{i}"))]
        )

    # every event landed on disk as its own object...
    event_files = glob.glob(
        str(tmp_path / "run" / "*.eval.sample" / "events" / "*.jsonl")
    )
    assert len(event_files) == n
    # ...and nothing is buffered in memory: only the counter advanced
    assert store._event_index[("1", 1)] == n
    assert store._pending_events == {}
    assert not hasattr(store, "_events")


def test_s3_round_trip(mock_s3):
    """The format writes to and reads from S3 (one PUT per object, no re-upload)."""
    task = Task(
        dataset=[Sample(input="hi", target="hello")],
        solver=generate(),
        scorer=includes(),
    )
    log = eval(
        task,
        model="mockllm/model",
        log_dir="s3://test-bucket/logs",
        log_format="eval.sample",
    )[0]
    assert log.location.startswith("s3://test-bucket/logs")

    re = read_eval_log(log.location)
    assert re.status == "success"
    assert re.samples is not None and len(re.samples) == 1
    assert re.samples[0].scores is not None

    # discoverable on S3 via the marker
    listed = list_eval_logs("s3://test-bucket/logs")
    assert any(li.type == "dir" for li in listed)


def test_compaction_to_eval_round_trip(tmp_path):
    """`inspect log convert` folds the directory into a .eval and recomputes aggregates."""
    from inspect_ai.log._convert import convert_eval_logs

    log = _run(tmp_path, n=2)
    src = read_eval_log(log.location)
    assert src.results is None  # raw folder carries no aggregates

    out_dir = str(tmp_path / "packed")
    convert_eval_logs(log.location, "eval", out_dir)
    eval_path = glob.glob(os.path.join(out_dir, "*.eval"))[0]

    packed = read_eval_log(eval_path)
    assert packed.status == src.status
    assert packed.samples is not None and src.samples is not None
    assert len(packed.samples) == len(src.samples)

    def key(s):
        return (str(s.id), s.epoch)

    for a, b in zip(sorted(src.samples, key=key), sorted(packed.samples, key=key)):
        assert a.id == b.id and a.epoch == b.epoch
        assert a.scores == b.scores
        assert len(a.messages) == len(b.messages)
        assert len(a.events) == len(b.events)

    # compaction recomputed the cross-run aggregates, matching a normal .eval run
    assert packed.results is not None and len(packed.results.scores) > 0
    baseline = eval(
        Task(
            dataset=[Sample(input=f"q{i}", target="hello") for i in range(2)],
            solver=generate(),
            scorer=includes(),
        ),
        model="mockllm/model",
        log_dir=str(tmp_path / "baseline"),
        log_format="eval",
    )[0]
    assert baseline.results is not None
    packed_metrics = {
        s.name: {m: v.value for m, v in s.metrics.items()}
        for s in packed.results.scores
    }
    base_metrics = {
        s.name: {m: v.value for m, v in s.metrics.items()}
        for s in baseline.results.scores
    }
    assert packed_metrics == base_metrics


def test_convert_eval_sample_via_cli_streaming(tmp_path):
    """Streaming convert also works (bounded memory during compaction)."""
    from inspect_ai.log._convert import convert_eval_logs

    log = _run(tmp_path, n=2)
    out_dir = str(tmp_path / "packed_stream")
    convert_eval_logs(log.location, "eval", out_dir, stream=True)
    eval_path = glob.glob(os.path.join(out_dir, "*.eval"))[0]
    packed = read_eval_log(eval_path)
    assert packed.samples is not None and len(packed.samples) == 2


def test_resume_host_state_from_dir(tmp_path):
    """M5: host-side resume state reconstructs from the .eval.sample dir alone.

    The hydrator consumes a HostContext (events / attachments / store /
    agent_state). Folding the directory yields exactly that — so resume needs
    only the dir plus a filesystem snapshot (the sandbox restic part).
    """
    from inspect_ai.log._recorders.eval_sample.resume import (
        host_context_from_sample_dir,
    )
    from inspect_ai.util._store import store_from_events

    log = _run(tmp_path, n=1)
    samples = read_eval_log(log.location).samples
    assert samples is not None
    sample = samples[0]

    ctx = host_context_from_sample_dir(log.location, id=sample.id, epoch=sample.epoch)

    # the transcript and attachments come straight from the directory
    assert len(ctx.condensed_events) == len(sample.events)
    assert ctx.attachments == sample.attachments
    # the store is folded from the same StoreEvents the live run produced
    assert ctx.store == dict(store_from_events(sample.events)._data)
    # no checkpoints fired in a plain eval, so no agent-state bag
    assert ctx.agent_state is None


def test_resume_agent_state_round_trip(tmp_path):
    """The cp.track() bag — the one host piece not in the stream — persists."""
    from inspect_ai.log._recorders.eval_sample.resume import (
        host_context_from_sample_dir,
    )
    from inspect_ai.log._recorders.eval_sample.store import SampleDirStore

    log = _run(tmp_path, n=1)
    samples = read_eval_log(log.location).samples
    assert samples is not None
    sample = samples[0]

    store = SampleDirStore(log.location, create=False)
    bag = {"messages": [{"role": "user", "content": "hi"}], "attempt": 3}
    store.write_agent_state(sample.id, sample.epoch, checkpoint_id=1, agent_state=bag)

    # readable directly and surfaced in the reconstructed HostContext
    assert store.read_agent_state(sample.id, sample.epoch) == bag
    ctx = host_context_from_sample_dir(
        log.location, id=sample.id, epoch=sample.epoch, checkpoint_id=1
    )
    assert ctx.agent_state == bag


def test_resume_to_checkpoint_boundary():
    """Folding can stop at any checkpoint — resume to a boundary, not just latest."""
    from inspect_ai.event._info import InfoEvent
    from inspect_ai.log._recorders.eval_sample.resume import _events_through_checkpoint

    def ckpt(n):
        from inspect_ai.event._checkpoint import CheckpointEvent

        e = CheckpointEvent.model_construct(checkpoint_id=n, event="checkpoint")
        return e

    events = [
        InfoEvent(source="t", data="a"),
        ckpt(1),
        InfoEvent(source="t", data="b"),
        ckpt(2),
        InfoEvent(source="t", data="c"),
    ]
    through_1 = _events_through_checkpoint(events, 1)
    assert len(through_1) == 2 and through_1[-1].checkpoint_id == 1
    through_2 = _events_through_checkpoint(events, 2)
    assert len(through_2) == 4 and through_2[-1].checkpoint_id == 2


def test_duckdb_queries_event_stream(tmp_path):
    """The cat-able jsonl events are directly queryable with DuckDB — no unzip."""
    duckdb = __import__("pytest").importorskip("duckdb")

    log = _run(tmp_path, n=2)
    pattern = os.path.join(log.location, "*.eval.sample", "events", "*.jsonl")

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT event, count(*) AS n FROM read_json_auto('{pattern}') "
        "GROUP BY event ORDER BY event"
    ).fetchall()
    by_type = {event: n for event, n in rows}
    assert by_type.get("sample_init", 0) == 2  # one per sample
    assert sum(by_type.values()) > 2


def test_viewer_buffer_api(tmp_path):
    """The viewer reads running + completed through the SampleBuffer interface."""
    log = _run(tmp_path, n=1)
    buf = sample_buffer(log.location)
    assert isinstance(buf, SampleDirStore)

    samples = buf.get_samples()
    assert samples is not None and samples != "NotModified"
    assert len(samples.samples) == 1
    # etag gating
    assert buf.get_samples(etag=samples.etag) == "NotModified"

    s0 = samples.samples[0]
    data = buf.get_sample_data(s0.id, s0.epoch)
    assert data is not None and len(data.events) > 0
    # incremental cursor returns only newer events
    cut = data.events[1].id
    after = buf.get_sample_data(s0.id, s0.epoch, after_event_id=cut)
    assert after is not None
    assert all(e.id > cut for e in after.events)
