"""Pydantic models for the on-disk checkpoint layout.

Defines the shape of the per-sample ``restic/restic-config.json`` and
the per-checkpoint ``ckpt-NNNNN.json`` checkpoint files. These are pure
data types — read/write helpers live with the write code.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from inspect_ai.model._model_output import ModelUsage

from .._triggers import CheckpointTriggerKind


class SnapshotDetails(BaseModel):
    """Per-backup stats captured in the checkpoint file.

    One per repo (host repo + one per active sandbox repo). Values come
    from restic's backup summary — see :class:`ResticBackupSummary`.
    """

    model_config = ConfigDict(extra="allow")

    snapshot_id: str
    """Restic snapshot id for this backup."""

    size_bytes: int
    """Bytes this snapshot added to its repo, after compression
    (restic's ``data_added_packed``)."""

    duration_ms: int
    """How long the restic invocation took, in milliseconds."""

    files: list[str] | None = None
    """Absolute paths of files added or changed in this snapshot (relative
    to its parent; the full file set for the first snapshot), capped at
    ``MAX_LISTED_FILES``."""

    additional_files: int | None = None
    """Count of files beyond ``MAX_LISTED_FILES`` not included in
    ``files``. ``None`` when nothing was truncated."""


class CheckpointUsage(BaseModel):
    """Sample usage as of this checkpoint, for continuation on resume."""

    model_config = ConfigDict(extra="allow")

    model_usage: dict[str, ModelUsage] = Field(default_factory=dict)
    """Per-model usage, from ``sample_model_usage()``."""

    role_usage: dict[str, ModelUsage] = Field(default_factory=dict)
    """Per-role usage, from ``sample_role_usage()``."""

    token_limit_usage: ModelUsage = Field(default_factory=ModelUsage)
    """The token limit node's own accumulator. Differs from the sum of
    ``model_usage`` by whatever was consumed under
    ``suspend_token_limit()``, which the limit ignores but the sample
    still accounts for."""

    cost: float = 0.0
    """Dollar cost recorded against the cost limit."""

    turns: int = 0
    """Turns recorded against the turn limit. Excludes turns taken under
    ``suspend_turn_limit()``."""

    time: float = 0.0
    """Seconds elapsed against the sample time limit."""

    working_time: float = 0.0
    """Seconds of working time (elapsed minus waiting) against the working
    limit."""


class Checkpoint(BaseModel):
    """Per-checkpoint metadata file (``<attempt>/ckpt-NNNNN.json``).

    Written atomically at each successful checkpoint. This file's
    existence is the commit point — the checkpoint is visible to
    resume only when this file is in place. See §1 and §4d.
    """

    model_config = ConfigDict(extra="allow")

    checkpoint_id: int
    """Ordinal integer (1, 2, 3, …) chosen by inspect at write time."""

    trigger: CheckpointTriggerKind
    """The policy that fired this checkpoint."""

    trigger_metadata: dict[str, JsonValue] | None = None
    """Trigger-specific fire details (e.g. configured threshold vs.
    actual usage at fire time)."""

    turn: int
    """Agent turn index at which this checkpoint was taken."""

    usage: CheckpointUsage | None = None
    """Sample usage as of this checkpoint. ``None`` for checkpoints written
    before this field existed, or fired with no live sample scope."""

    created_at: datetime
    """When the checkpoint was committed."""

    duration_ms: int
    """How long the checkpoint cycle took, in milliseconds."""

    size_bytes: int
    """Total on-disk size added by this checkpoint (sum of host + sandboxes)."""

    host: SnapshotDetails
    """Stats for the host repo backup this cycle."""

    sandboxes: dict[str, SnapshotDetails] = Field(default_factory=dict)
    """Per-sandbox stats keyed by sandbox name. Empty when checkpointing is
    host-only."""


class ResticConfig(BaseModel):
    """Per-sample restic config file (``<sample-root>/restic/restic-config.json``).

    Lives alongside the per-sample restic repos under ``restic/``.
    Written once at first checkpoint setup for a sample; never
    rewritten. Preserved across retries of the same sample via the FS
    copy at resume — so the same password unlocks the FS-copied
    ``host/`` and ``sandboxes/<name>/`` repos in the new sample dir.
    """

    model_config = ConfigDict(extra="allow")

    restic_password: str
    """Password used by every repo (host + each sandbox) under this
    sample. Reaches sandbox-side restic via the per-exec environment;
    never persisted in the sandbox."""
