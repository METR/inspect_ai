"""Tests for the sample checkpoints dir, restic-config.json, and checkpoint file writes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pytest

from inspect_ai.util._checkpoint._layout.sample_checkpoints_dir import (
    _read_restic_config,
    ensure_restic_config,
    ensure_sample_checkpoints_dir,
    sample_checkpoints_dir,
    scan_latest_committed_checkpoint,
    write_checkpoint_file,
)
from inspect_ai.util._checkpoint._layout.schemas import (
    Checkpoint,
    CheckpointUsage,
    ResticConfig,
    SnapshotDetails,
)
from inspect_ai.util._checkpoint._triggers import CheckpointTriggerKind
from inspect_ai.util._limit import Limit


def _info(
    snapshot_id: str, size_bytes: int = 0, duration_ms: int = 0
) -> SnapshotDetails:
    return SnapshotDetails(
        snapshot_id=snapshot_id, size_bytes=size_bytes, duration_ms=duration_ms
    )


def _checkpoint(
    *,
    checkpoint_id: int,
    trigger: CheckpointTriggerKind,
    turn: int,
    host: SnapshotDetails,
    sandboxes: dict[str, SnapshotDetails] | None = None,
    duration_ms: int = 0,
) -> Checkpoint:
    sb = sandboxes or {}
    return Checkpoint(
        checkpoint_id=checkpoint_id,
        trigger=trigger,
        turn=turn,
        created_at=datetime.now(timezone.utc),
        duration_ms=duration_ms,
        size_bytes=host.size_bytes + sum(s.size_bytes for s in sb.values()),
        host=host,
        sandboxes=sb,
    )


def test_sample_checkpoints_dir_uses_sample_id_and_epoch() -> None:
    assert (
        sample_checkpoints_dir("/logs/foo.checkpoints", "sample-7", 0)
        == "/logs/foo.checkpoints/sample-7__0"
    )


def test_sample_checkpoints_dir_accepts_int_sample_id() -> None:
    assert (
        sample_checkpoints_dir("/logs/foo.checkpoints", 42, 1)
        == "/logs/foo.checkpoints/42__1"
    )


async def test_ensure_creates_dir_and_returns_path(tmp_path: Path) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    sample_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    assert Path(sample_dir).is_dir()
    assert sample_dir == f"{eval_dir}/s1__0"


async def test_ensure_is_idempotent(tmp_path: Path) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    a = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    b = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    assert a == b
    assert Path(a).is_dir()


async def test_ensure_creates_parent_eval_dir(tmp_path: Path) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    assert Path(eval_dir).is_dir()


async def test_ensure_restic_config_mints_password_on_first_call(
    tmp_path: Path,
) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    sample_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    sample = await ensure_restic_config(sample_dir)
    assert sample.restic_password
    assert (Path(sample_dir) / "restic" / "restic-config.json").is_file()


async def test_ensure_restic_config_preserves_password_on_second_call(
    tmp_path: Path,
) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    sample_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    first = await ensure_restic_config(sample_dir)
    second = await ensure_restic_config(sample_dir)
    assert first.restic_password == second.restic_password


async def test_ensure_restic_config_different_samples_get_distinct_passwords(
    tmp_path: Path,
) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    a_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    b_dir = await ensure_sample_checkpoints_dir(eval_dir, "s2", 0)
    a = await ensure_restic_config(a_dir)
    b = await ensure_restic_config(b_dir)
    assert a.restic_password != b.restic_password


async def test_read_restic_config_returns_written_value(tmp_path: Path) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    sample_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    written = await ensure_restic_config(sample_dir)
    read = await _read_restic_config(sample_dir)
    assert read.restic_password == written.restic_password


async def test_restic_config_round_trip_pydantic(tmp_path: Path) -> None:
    eval_dir = str(tmp_path / "foo.checkpoints")
    sample_dir = await ensure_sample_checkpoints_dir(eval_dir, "s1", 0)
    await ensure_restic_config(sample_dir)
    raw = (Path(sample_dir) / "restic" / "restic-config.json").read_text()
    parsed = ResticConfig.model_validate_json(raw)
    assert parsed.restic_password


async def test_write_checkpoint_file_returns_zero_padded_path(tmp_path: Path) -> None:
    sample_dir = await ensure_sample_checkpoints_dir(
        str(tmp_path / "foo.checkpoints"), "s1", 0
    )
    path = await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=_checkpoint(
            checkpoint_id=1,
            trigger="turn",
            turn=3,
            host=_info("snap-1"),
        ),
    )
    assert path == f"{sample_dir}/ckpt-00001.json"
    assert Path(path).is_file()


async def test_checkpoint_file_contents_round_trip(tmp_path: Path) -> None:
    sample_dir = await ensure_sample_checkpoints_dir(
        str(tmp_path / "foo.checkpoints"), "s", 0
    )
    path = await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=_checkpoint(
            checkpoint_id=42,
            trigger="manual",
            turn=7,
            host=_info("snap-42", size_bytes=1000, duration_ms=10),
            sandboxes={"default": _info("sb-42", size_bytes=234, duration_ms=20)},
            duration_ms=99,
        ),
    )
    checkpoint = Checkpoint.model_validate_json(Path(path).read_text())
    assert checkpoint.checkpoint_id == 42
    assert checkpoint.trigger == "manual"
    assert checkpoint.turn == 7
    assert checkpoint.host.snapshot_id == "snap-42"
    assert checkpoint.host.duration_ms == 10
    assert checkpoint.sandboxes["default"].snapshot_id == "sb-42"
    assert checkpoint.size_bytes == 1234  # rolled-up total
    assert checkpoint.duration_ms == 99  # whole-cycle


async def test_checkpoint_file_filename_zero_padded_for_lexical_sort(
    tmp_path: Path,
) -> None:
    sample_dir = await ensure_sample_checkpoints_dir(
        str(tmp_path / "foo.checkpoints"), "s", 0
    )
    paths = [
        await write_checkpoint_file(
            sample_checkpoints_dir=sample_dir,
            checkpoint=_checkpoint(
                checkpoint_id=cid,
                trigger="turn",
                turn=cid,
                host=_info(f"snap-{cid}"),
            ),
        )
        for cid in (1, 2, 10, 100)
    ]
    names = [Path(p).name for p in paths]
    assert names == sorted(names)
    assert names == [
        "ckpt-00001.json",
        "ckpt-00002.json",
        "ckpt-00010.json",
        "ckpt-00100.json",
    ]


async def test_checkpoint_file_is_pretty_printed_json(tmp_path: Path) -> None:
    sample_dir = await ensure_sample_checkpoints_dir(
        str(tmp_path / "foo.checkpoints"), "s", 0
    )
    path = await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=_checkpoint(
            checkpoint_id=1,
            trigger="turn",
            turn=1,
            host=_info("snap-1"),
        ),
    )
    raw = Path(path).read_text()
    assert json.loads(raw)["checkpoint_id"] == 1
    assert "\n" in raw


async def test_scan_latest_committed_checkpoint_returns_latest_parseable(
    tmp_path: Path,
) -> None:
    sample_dir = await ensure_sample_checkpoints_dir(
        str(tmp_path / "foo.checkpoints"), "s", 0
    )
    await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=_checkpoint(
            checkpoint_id=1,
            trigger="turn",
            turn=1,
            host=_info("snap-1"),
        ),
    )
    await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=_checkpoint(
            checkpoint_id=2,
            trigger="agent_complete",
            turn=2,
            host=_info("snap-2"),
        ),
    )
    (Path(sample_dir) / "ckpt-00003.json").write_text("{")

    checkpoint = await scan_latest_committed_checkpoint(sample_dir)

    assert checkpoint is not None
    assert checkpoint.checkpoint_id == 2
    assert checkpoint.trigger == "agent_complete"


async def test_resume_carries_usage(tmp_path: Path) -> None:
    from inspect_ai._eval.task.run import _resume_if_checkpointed

    eval_dir = tmp_path / "logs.checkpoints"
    sample_dir = sample_checkpoints_dir(str(eval_dir), 1, 1)
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    await write_checkpoint_file(
        sample_checkpoints_dir=sample_dir,
        checkpoint=Checkpoint(
            checkpoint_id=1,
            trigger="turn",
            turn=2,
            created_at=datetime.now(timezone.utc),
            duration_ms=1,
            size_bytes=1,
            host=SnapshotDetails(snapshot_id="a", size_bytes=1, duration_ms=1),
            usage=CheckpointUsage(cost=0.5, turns=2, time=30.0, working_time=25.0),
        ),
    )

    resume = await _resume_if_checkpointed(str(eval_dir), 1, 1)

    assert resume is not None
    assert resume.usage is not None
    assert resume.usage.cost == 0.5
    assert resume.usage.turns == 2


def _usage_seed() -> CheckpointUsage:
    from inspect_ai.model._model_output import ModelUsage

    return CheckpointUsage(
        model_usage={"mockllm/model": ModelUsage(total_tokens=40)},
        token_limit_usage=ModelUsage(total_tokens=40),
        cost=0.4,
        turns=4,
        time=30.0,
        working_time=25.0,
    )


def test_prior_usage_seeds_are_applied_when_enabled() -> None:
    from inspect_ai._eval.task.run import _prior_usage
    from inspect_ai.util._checkpoint._triggers import Manual
    from inspect_ai.util._checkpoint.checkpointer import ResumeCheckpoint
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig

    resume = ResumeCheckpoint(
        sample_checkpoints_dir="/tmp/x", attempt="resume", usage=_usage_seed()
    )
    config = ResolvedCheckpointConfig(trigger=Manual(), restore_usage=True)

    assert _prior_usage(resume, config) is not None


def test_prior_usage_is_none_when_flag_off() -> None:
    from inspect_ai._eval.task.run import _prior_usage
    from inspect_ai.util._checkpoint._triggers import Manual
    from inspect_ai.util._checkpoint.checkpointer import ResumeCheckpoint
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig

    resume = ResumeCheckpoint(
        sample_checkpoints_dir="/tmp/x", attempt="resume", usage=_usage_seed()
    )
    config = ResolvedCheckpointConfig(trigger=Manual())

    assert _prior_usage(resume, config) is None


def test_prior_usage_is_none_without_a_resume() -> None:
    from inspect_ai._eval.task.run import _prior_usage
    from inspect_ai.util._checkpoint._triggers import Manual
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig

    config = ResolvedCheckpointConfig(trigger=Manual(), restore_usage=True)

    assert _prior_usage(None, config) is None


class _GuardSeed(NamedTuple):
    token_usage: int = 0
    cost_usage: float = 0.0
    turns: int = 0
    time_usage: float = 0.0
    working_usage: float = 0.0


class _GuardNodes(NamedTuple):
    token: Limit
    cost: Limit
    turn: Limit
    time: Limit
    working: Limit


def _seeded_nodes(seed: _GuardSeed, *, unlimited: bool = False) -> _GuardNodes:
    """The five sample limit nodes, seeded with `seed` as a resume would."""
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._limit import (
        cost_limit,
        seed_limit_usage,
        time_limit,
        token_limit,
        turn_limit,
        working_limit,
    )

    nodes = _GuardNodes(
        token=token_limit(None if unlimited else 100),
        cost=cost_limit(None if unlimited else 1.0),
        turn=turn_limit(None if unlimited else 10),
        time=time_limit(None if unlimited else 60.0),
        working=working_limit(None if unlimited else 50.0),
    )
    seed_limit_usage(
        token=nodes.token,
        cost=nodes.cost,
        turn=nodes.turn,
        time=nodes.time,
        working=nodes.working,
        token_usage=ModelUsage(total_tokens=seed.token_usage),
        cost_usage=seed.cost_usage,
        turns=seed.turns,
        time_usage=seed.time_usage,
        working_usage=seed.working_usage,
    )
    return nodes


def _check_exhausted(nodes: _GuardNodes) -> None:
    from inspect_ai._eval.task.run import _raise_if_prior_usage_exhausted

    _raise_if_prior_usage_exhausted(
        token=nodes.token,
        cost=nodes.cost,
        turn=nodes.turn,
        time=nodes.time,
        working=nodes.working,
    )


def test_prior_usage_below_every_ceiling_does_not_raise() -> None:
    _check_exhausted(
        _seeded_nodes(
            _GuardSeed(
                token_usage=99,
                cost_usage=0.99,
                turns=9,
                time_usage=59.0,
                working_usage=49.0,
            )
        )
    )


@pytest.mark.parametrize(
    "seed,expected_type",
    [
        (_GuardSeed(token_usage=100), "token"),
        (_GuardSeed(cost_usage=1.0), "cost"),
        (_GuardSeed(turns=10), "turn"),
        (_GuardSeed(time_usage=60.0), "time"),
        (_GuardSeed(working_usage=50.0), "working"),
    ],
)
def test_prior_usage_at_a_ceiling_fails_the_resume(
    seed: _GuardSeed, expected_type: str
) -> None:
    from inspect_ai.util._limit import LimitExceededError

    with pytest.raises(LimitExceededError) as exc_info:
        _check_exhausted(_seeded_nodes(seed))

    assert exc_info.value.type == expected_type
    assert "Restored usage from checkpoint" in str(exc_info.value)


def test_prior_usage_against_unlimited_nodes_does_not_raise() -> None:
    _check_exhausted(
        _seeded_nodes(
            _GuardSeed(
                token_usage=10_000,
                cost_usage=100.0,
                turns=1_000,
                time_usage=10_000.0,
                working_usage=10_000.0,
            ),
            unlimited=True,
        )
    )


def test_seeded_model_usage_is_isolated_from_the_checkpoint() -> None:
    """A seeded accumulator must not alias the checkpoint's own usage.

    An error retry re-seeds from the same `ResumeCheckpoint`, so an alias
    would fold the abandoned attempt's tokens into the next seed.
    """
    from inspect_ai.model._model import (
        init_sample_model_data,
        sample_model_usage,
        sample_model_usage_context_var,
        sample_role_usage,
        sample_role_usage_context_var,
    )
    from inspect_ai.model._model_output import ModelUsage

    seed = _usage_seed()
    seed.role_usage = {"grader": ModelUsage(total_tokens=2)}

    model_token = sample_model_usage_context_var.set({})
    role_token = sample_role_usage_context_var.set({})
    try:
        init_sample_model_data(seed.model_usage, seed.role_usage)

        assert sample_model_usage()["mockllm/model"].total_tokens == 40
        assert sample_role_usage()["grader"].total_tokens == 2

        sample_model_usage()["mockllm/model"].total_tokens = 999999
        sample_role_usage()["grader"].total_tokens = 999999
        sample_model_usage()["other/model"] = ModelUsage(total_tokens=1)

        assert seed.model_usage == {"mockllm/model": ModelUsage(total_tokens=40)}
        assert seed.role_usage == {"grader": ModelUsage(total_tokens=2)}
    finally:
        sample_model_usage_context_var.reset(model_token)
        sample_role_usage_context_var.reset(role_token)


def test_unseeded_model_usage_starts_empty() -> None:
    from inspect_ai.model._model import (
        init_sample_model_data,
        sample_model_usage,
        sample_model_usage_context_var,
        sample_role_usage,
        sample_role_usage_context_var,
    )
    from inspect_ai.model._model_output import ModelUsage

    model_token = sample_model_usage_context_var.set(
        {"mockllm/model": ModelUsage(total_tokens=40)}
    )
    role_token = sample_role_usage_context_var.set(
        {"grader": ModelUsage(total_tokens=2)}
    )
    try:
        init_sample_model_data()

        assert sample_model_usage() == {}
        assert sample_role_usage() == {}
    finally:
        sample_model_usage_context_var.reset(model_token)
        sample_role_usage_context_var.reset(role_token)
