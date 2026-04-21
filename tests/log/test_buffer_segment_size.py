import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from inspect_ai.log._log import EvalSampleSummary
from inspect_ai.log._recorders.buffer.filestore import (
    Manifest,
    SampleBufferFilestore,
    SampleManifest,
    Segment,
    SegmentFile,
    segment_name,
    segments_for_sample_cursor,
)
from inspect_ai.log._recorders.buffer.types import EventData, SampleData


@pytest.fixture
def buffer_location() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "run.eval")


def _segment_file(sample_id: str = "s1", epoch: int = 0) -> SegmentFile:
    data = SampleData(
        events=[
            EventData(
                id=1,
                event_id="e1",
                sample_id=sample_id,
                epoch=epoch,
                event={"kind": "hello"},
            )
        ],
        attachments=[],
    )
    return SegmentFile(id=sample_id, epoch=epoch, data=data)


def test_write_segment_returns_actual_zip_size(buffer_location: str) -> None:
    store = SampleBufferFilestore(buffer_location, create=True)
    returned_size = store.write_segment(0, [_segment_file()])
    seg_path = f"{store._dir}{segment_name(0)}"
    assert returned_size == os.path.getsize(seg_path)
    assert returned_size > 0


def test_segment_size_field_is_optional_for_back_compat() -> None:
    manifest_json = (
        '{"metrics": [], "samples": [], "segments": ['
        '{"id": 0, "last_event_id": 0, "last_attachment_id": 0}'
        "]}"
    )
    m = Manifest.model_validate_json(manifest_json)
    assert m.segments[0].size is None


def test_segment_size_survives_manifest_roundtrip() -> None:
    from inspect_ai.log._recorders.buffer.filestore import Segment

    seg = Segment(id=3, last_event_id=10, last_attachment_id=5, size=12345)
    dumped = seg.model_dump_json()
    rehydrated = Segment.model_validate_json(dumped)
    assert rehydrated.size == 12345


def test_sync_to_filestore_populates_segment_size() -> None:
    """End-to-end: sync writes a manifest where every Segment has the correct size.

    database → sync_to_filestore writes a manifest where every Segment has `size`
    populated with the actual zip byte length.
    """
    from inspect_ai.event._info import InfoEvent
    from inspect_ai.log._recorders.buffer.database import (
        SampleBufferDatabase,
        sync_to_filestore,
    )
    from inspect_ai.log._recorders.types import SampleEvent

    with tempfile.TemporaryDirectory() as tmpdir:
        db_dir = Path(tmpdir)
        db = SampleBufferDatabase(location="testdb", create=True, db_dir=db_dir)
        filestore_dir = Path(tmpdir) / "fs"
        filestore = SampleBufferFilestore(str(filestore_dir), create=True)

        sample = EvalSampleSummary(id="s1", epoch=0, input="a", target="b")
        db.start_sample(sample)
        db.log_events([SampleEvent(id="s1", epoch=0, event=InfoEvent(data="hello"))])

        sync_to_filestore(db, filestore)

        manifest = filestore.read_manifest()
        assert manifest is not None
        assert len(manifest.segments) >= 1
        for seg in manifest.segments:
            seg_path = f"{filestore._dir}{segment_name(seg.id)}"
            assert seg.size == os.path.getsize(seg_path)


def _manifest(segs: list[Segment], sample_segs: list[int]) -> Manifest:
    return Manifest(
        metrics=[],
        samples=[
            SampleManifest(
                summary=EvalSampleSummary(id="s", epoch=0, input="i", target="t"),
                segments=sample_segs,
            )
        ],
        segments=segs,
    )


def test_segments_for_sample_cursor_returns_all_when_cursor_is_minus_one() -> None:
    segs = [Segment(id=i, last_event_id=i, last_attachment_id=i) for i in range(3)]
    m = _manifest(segs, [0, 1, 2])
    sample = m.samples[0]
    out = segments_for_sample_cursor(
        m,
        sample,
        after_event_id=-1,
        after_attachment_id=-1,
        after_message_pool_id=-1,
        after_call_pool_id=-1,
    )
    assert [s.id for s in out] == [0, 1, 2]


def test_segments_for_sample_cursor_prunes_by_event_id() -> None:
    # All cursors set high enough that only the event-id dimension gates.
    segs = [
        Segment(id=0, last_event_id=5, last_attachment_id=0),
        Segment(id=1, last_event_id=10, last_attachment_id=0),
        Segment(id=2, last_event_id=15, last_attachment_id=0),
    ]
    m = _manifest(segs, [0, 1, 2])
    sample = m.samples[0]
    out = segments_for_sample_cursor(
        m,
        sample,
        after_event_id=10,
        after_attachment_id=100,
        after_message_pool_id=100,
        after_call_pool_id=100,
    )
    # Only segment 2 has last_event_id > 10; others have no dimension above
    # any cursor so the OR-filter excludes them.
    assert [s.id for s in out] == [2]


def test_segments_for_sample_cursor_or_logic_across_cursor_types() -> None:
    # Segment 0 qualifies via the attachment dimension only; segment 1 has
    # no dimension above any cursor and is excluded.
    segs = [
        Segment(id=0, last_event_id=5, last_attachment_id=100),
        Segment(id=1, last_event_id=5, last_attachment_id=5),
    ]
    m = _manifest(segs, [0, 1])
    sample = m.samples[0]
    out = segments_for_sample_cursor(
        m,
        sample,
        after_event_id=10,
        after_attachment_id=50,
        after_message_pool_id=50,
        after_call_pool_id=50,
    )
    assert [s.id for s in out] == [0]


def test_segments_for_sample_cursor_ignores_segments_not_in_sample() -> None:
    segs = [Segment(id=i, last_event_id=i, last_attachment_id=i) for i in range(3)]
    m = _manifest(segs, [0, 2])  # sample excludes segment 1
    sample = m.samples[0]
    out = segments_for_sample_cursor(
        m,
        sample,
        after_event_id=-1,
        after_attachment_id=-1,
        after_message_pool_id=-1,
        after_call_pool_id=-1,
    )
    assert [s.id for s in out] == [0, 2]
