"""Tests for the on-disk layout types.

Pure data types: `Checkpoint` (per-checkpoint `ckpt-NNNNN.json`)
and `ResticConfig` (per-sample `restic/restic-config.json`).
Verifies the contract: required fields, validation strictness on
triggers, JSON round-trip, and ``extra="allow"`` forward-compat.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from inspect_ai.util._checkpoint._layout.schemas import Checkpoint, ResticConfig


def _info(
    snapshot_id: str, size_bytes: int = 0, duration_ms: int = 0
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot_id,
        "size_bytes": size_bytes,
        "duration_ms": duration_ms,
    }


_BASE_CHECKPOINT = {
    "checkpoint_id": 1,
    "trigger": "time",
    "turn": 42,
    "created_at": "2026-04-26T14:23:11Z",
    "duration_ms": 842,
    "size_bytes": 1834291,
    "host": _info("abc123"),
    "sandboxes": {
        "default": _info("def456"),
        "tools": _info("ghi789"),
    },
}

_BASE_SAMPLE = {
    "restic_password": "s3cr3t",
}


# --- Checkpoint -----------------------------------------------------------


def test_checkpoint_basic_round_trip() -> None:
    checkpoint = Checkpoint.model_validate(_BASE_CHECKPOINT)
    assert checkpoint.checkpoint_id == 1
    assert checkpoint.trigger == "time"
    assert checkpoint.created_at == datetime(
        2026, 4, 26, 14, 23, 11, tzinfo=timezone.utc
    )
    assert checkpoint.host.snapshot_id == "abc123"
    assert {name: info.snapshot_id for name, info in checkpoint.sandboxes.items()} == {
        "default": "def456",
        "tools": "ghi789",
    }

    # JSON round-trip preserves all fields.
    rehydrated = Checkpoint.model_validate_json(checkpoint.model_dump_json())
    assert rehydrated == checkpoint


@pytest.mark.parametrize("trigger", ["time", "turn", "manual", "agent_complete"])
def test_checkpoint_accepts_all_documented_triggers(trigger: str) -> None:
    payload = {**_BASE_CHECKPOINT, "trigger": trigger}
    checkpoint = Checkpoint.model_validate(payload)
    assert checkpoint.trigger == trigger


def test_checkpoint_rejects_unknown_trigger() -> None:
    payload = {**_BASE_CHECKPOINT, "trigger": "bogus"}
    with pytest.raises(ValidationError):
        Checkpoint.model_validate(payload)


def test_checkpoint_extra_fields_preserved_round_trip() -> None:
    """Forward-compat for unknown fields.

    A future inspect adds a field; older inspect must keep that field
    intact when round-tripping the JSON.
    """
    payload = {**_BASE_CHECKPOINT, "future_field": {"nested": [1, 2, 3]}}
    checkpoint = Checkpoint.model_validate(payload)
    dumped = json.loads(checkpoint.model_dump_json())
    assert dumped["future_field"] == {"nested": [1, 2, 3]}


def test_checkpoint_empty_sandboxes_is_valid() -> None:
    """Host-only checkpointing case: no sandbox snapshots."""
    payload = {**_BASE_CHECKPOINT, "sandboxes": {}}
    checkpoint = Checkpoint.model_validate(payload)
    assert checkpoint.sandboxes == {}


# --- SnapshotDetails.files (opt-in file listing) ----------------------


def test_snapshot_files_round_trip() -> None:
    """``files`` / ``additional_files`` survive a JSON round-trip."""
    payload = {
        **_BASE_CHECKPOINT,
        "sandboxes": {
            "default": {
                **_info("def456"),
                "files": ["/root/a.txt", "/root/b.txt"],
                "additional_files": 23,
            }
        },
    }
    checkpoint = Checkpoint.model_validate(payload)
    details = checkpoint.sandboxes["default"]
    assert details.files == ["/root/a.txt", "/root/b.txt"]
    assert details.additional_files == 23

    rehydrated = Checkpoint.model_validate_json(checkpoint.model_dump_json())
    assert rehydrated == checkpoint


def test_snapshot_files_omitted_when_unset() -> None:
    """Disabled (None) listing → keys absent (writer dumps exclude_none)."""
    checkpoint = Checkpoint.model_validate(_BASE_CHECKPOINT)
    dumped = json.loads(checkpoint.model_dump_json(exclude_none=True))
    default = dumped["sandboxes"]["default"]
    assert "files" not in default
    assert "additional_files" not in default


def test_snapshot_additional_files_omitted_when_not_truncated() -> None:
    """``files`` present, ``additional_files`` omitted when not truncated."""
    payload = {
        **_BASE_CHECKPOINT,
        "sandboxes": {"default": {**_info("def456"), "files": ["/root/a.txt"]}},
    }
    checkpoint = Checkpoint.model_validate(payload)
    default = json.loads(checkpoint.model_dump_json(exclude_none=True))["sandboxes"][
        "default"
    ]
    assert default["files"] == ["/root/a.txt"]
    assert "additional_files" not in default


# --- ResticConfig -----------------------------------------------------


def test_sample_basic_round_trip() -> None:
    sample = ResticConfig.model_validate(_BASE_SAMPLE)
    assert sample.restic_password == "s3cr3t"

    rehydrated = ResticConfig.model_validate_json(sample.model_dump_json())
    assert rehydrated == sample


def test_sample_requires_password() -> None:
    with pytest.raises(ValidationError):
        ResticConfig.model_validate({})


def test_sample_extra_fields_preserved_round_trip() -> None:
    payload = {**_BASE_SAMPLE, "future_field": "later"}
    sample = ResticConfig.model_validate(payload)
    dumped = json.loads(sample.model_dump_json())
    assert dumped["future_field"] == "later"


def test_usage_round_trips() -> None:
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._checkpoint._layout.schemas import CheckpointUsage

    usage = CheckpointUsage(
        model_usage={
            "openai/gpt-5": ModelUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            )
        },
        role_usage={},
        token_limit_usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        cost=0.25,
        turns=3,
        time=42.5,
        working_time=40.0,
    )

    assert CheckpointUsage.model_validate_json(usage.model_dump_json()) == usage


def test_usage_defaults_to_zero() -> None:
    from inspect_ai.util._checkpoint._layout.schemas import CheckpointUsage

    usage = CheckpointUsage()

    assert usage.model_usage == {}
    assert usage.cost == 0.0
    assert usage.turns == 0
    assert usage.time == 0.0
    assert usage.working_time == 0.0


def test_checkpoint_without_usage_parses() -> None:
    checkpoint = Checkpoint.model_validate(
        {
            "checkpoint_id": 1,
            "trigger": "turn",
            "turn": 2,
            "created_at": datetime(2026, 8, 19, tzinfo=timezone.utc),
            "duration_ms": 100,
            "size_bytes": 10,
            "host": _info("abc"),
        }
    )

    assert checkpoint.usage is None
