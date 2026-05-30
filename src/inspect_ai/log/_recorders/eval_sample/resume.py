"""Reconstruct sandbox-agent resume state from a `.eval.sample` directory.

The checkpoint subsystem (`util/_checkpoint/`) hydrates a resumed sample from a
``HostContext`` — transcript events, attachments, store, and the ``cp.track()``
agent-state bag — plus restic snapshots of the sandbox filesystem. Comparing
that against what a `.eval.sample` directory already holds:

- events / attachments — the directory IS the event stream
- store — derivable by folding ``StoreEvent``s (``store_from_events``)
- agent_state — the one host piece not in the stream; persisted alongside the
  events under ``checkpoints/<id>/agent_state.jsonl``

So the directory is the single source of truth for host-side resume state: this
module folds it back into the exact ``HostContext`` the hydrator consumes, for
any checkpoint boundary. Only the sandbox filesystem still comes from restic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_ai.event._checkpoint import CheckpointEvent
from inspect_ai.event._event import Event

from ..._recover._reconstruct import _deserialize_events, collapse_event_versions
from ..buffer.types import SampleData
from .store import SampleDirStore

if TYPE_CHECKING:
    from inspect_ai.util._checkpoint._layout.host_context import HostContext


def _events_through_checkpoint(events: list[Event], checkpoint_id: int) -> list[Event]:
    """Events up to and including the Nth checkpoint — resume to any boundary."""
    out: list[Event] = []
    for event in events:
        out.append(event)
        if isinstance(event, CheckpointEvent) and event.checkpoint_id == checkpoint_id:
            return out
    return out


def latest_checkpoint_id_from_sample_dir(
    location: str, id: str | int | None = None, epoch: int = 1
) -> int | None:
    """Highest committed checkpoint id in the sample's stream, or ``None``.

    For `.eval.sample` the ``CheckpointEvent`` in the event stream IS the
    checkpoint commit marker — there is no separate checkpoint file — so a
    sample has a resumable checkpoint iff its stream contains one. Reads raw
    event dicts (no deserialization) so it stays cheap.
    """
    store = SampleDirStore(location, create=False)
    if id is None:
        summaries = store.sample_summaries()
        if not summaries:
            return None
        id, epoch = summaries[0].id, summaries[0].epoch
    data = store.get_sample_data(id, epoch)
    if data is None:
        return None
    ids = [
        cid
        for e in data.events
        if isinstance(e.event, dict) and e.event.get("event") == "checkpoint"
        for cid in [e.event.get("checkpoint_id")]
        if isinstance(cid, int)
    ]
    return max(ids) if ids else None


def host_context_from_sample_dir(
    location: str,
    id: str | int | None = None,
    epoch: int = 1,
    checkpoint_id: int | None = None,
) -> "HostContext":
    """Fold a `.eval.sample` sample into the `HostContext` the hydrator consumes.

    Args:
        location: The eval folder or a lone `.eval.sample` directory.
        id: Sample id (defaults to the first sample found).
        epoch: Sample epoch.
        checkpoint_id: Resume boundary — fold events up to this checkpoint.
            ``None`` uses the full stream (latest state).

    Returns:
        A ``HostContext`` with events/attachments/store/agent_state populated
        from the directory.
    """
    # imported lazily — the checkpoint layout pulls in the log/model chain
    from inspect_ai.util._checkpoint._layout.host_context import HostContext
    from inspect_ai.util._store import store_from_events

    store = SampleDirStore(location, create=False)
    if id is None:
        summaries = store.sample_summaries()
        if not summaries:
            raise IndexError(f"No samples found in {location}")
        id, epoch = summaries[0].id, summaries[0].epoch

    data = store.get_sample_data(id, epoch) or SampleData(events=[], attachments=[])
    events = _deserialize_events(
        [e.event for e in collapse_event_versions(data.events)]
    )
    if checkpoint_id is not None:
        events = _events_through_checkpoint(events, checkpoint_id)

    return HostContext(
        condensed_events=events,
        msg_pool=[],
        call_pool=[],
        attachments={a.hash: a.content for a in data.attachments},
        store=dict(store_from_events(events)._data),
        agent_state=store.read_agent_state(id, epoch, checkpoint_id),
    )
