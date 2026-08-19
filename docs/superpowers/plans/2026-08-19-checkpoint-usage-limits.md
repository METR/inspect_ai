# Checkpoint Usage Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a checkpoint-resumed sample continue its token / cost / turn / time / working-time budget instead of restarting it at zero, behind an opt-in `restore_usage` flag that defaults off.

**Architecture:** Each checkpoint fire writes a `CheckpointUsage` block into `ckpt-NNNNN.json`. That file is already parsed before the sample runs, so on an opted-in resume the values are applied as construction-time *seeds* on the limit nodes and the sample usage contextvars — no mid-flight mutation of running limits. Capture is unconditional; only restoration is gated by the flag.

**Tech Stack:** Python 3.11+, pydantic v2, anyio, pytest + anyio (dual asyncio/trio), ruff, mypy.

**Spec:** `design/checkpoint-usage-limits.md`. Read it before Task 1 — it carries the *why* behind several non-obvious choices this plan only states.

## Global Constraints

- **Comments: terse, plain language, only when non-obvious.** Do not restate what a name already says. Do not paraphrase the spec or a commit message into a comment. A comment earns its place only when a reader would otherwise get it wrong — an ordering constraint, a divergence between two similar-looking values, a guard whose absence looks like an oversight. Prefer no comment. Per `AGENTS.md`, document rationale in the function's docstring, not at the call site.
- **Types: strict.** Every function annotated. Never `typing.Any`, never `# type: ignore`.
- **Async tests:** `async def test_...` only — no `@pytest.mark.asyncio` (blocked by `tests/conftest.py`). Use `anyio.sleep`, not `asyncio.sleep`.
- **Public API is frozen.** `token_limit`, `cost_limit`, `turn_limit`, `time_limit`, `working_limit` (exported from `inspect_ai.util`) and `TaskState.__init__` must not gain parameters. Seeding is internal.
- **Default is off.** `restore_usage` resolves to `False`. Every behaviour change in Tasks 5–8 must be unreachable without it.
- Run `ruff format`, `ruff check --fix`, and `mypy --exclude tests/test_package src tests` before each commit.
- Commit at the end of every task. Never change the `src/inspect_ai/_view/ts-mono` gitlink except in Task 9.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/inspect_ai/util/_checkpoint/_layout/schemas.py` | `CheckpointUsage` model + `Checkpoint.usage` field | 1 |
| `src/inspect_ai/util/_checkpoint/config.py` | `restore_usage` on the config layers + merge | 2 |
| `src/inspect_ai/util/_checkpoint/parse_cli.py` | `restore_usage` in the YAML model | 2 |
| `src/inspect_ai/util/_limit.py` | Private seeding entry point + per-node seed support | 3 |
| `src/inspect_ai/_util/working.py` | `SampleTiming` prior-usage offsets | 4 |
| `src/inspect_ai/util/_checkpoint/checkpointer_impl.py` | Capture usage at fire time | 5 |
| `src/inspect_ai/util/_checkpoint/checkpointer.py` | `ResumeCheckpoint.usage` | 6 |
| `src/inspect_ai/_eval/task/run.py` | Carry usage onto `ResumeCheckpoint`; apply seeds | 6, 7 |
| `src/inspect_ai/model/_model.py` | Seedable `init_sample_model_data()` | 7 |
| `docs/checkpointing.qmd`, `CHANGELOG.md` | User-facing docs | 8 |
| ts-mono submodule + `inspect-openapi.json` | Regenerated types | 9 |

Tasks 1–4 are independent leaves and can be done in any order. Task 5 needs 1. Task 6 needs 1. Task 7 needs 2, 3, 4, 6. Task 8 needs 7. Task 9 needs 1.

---

### Task 1: `CheckpointUsage` schema

**Files:**
- Modify: `src/inspect_ai/util/_checkpoint/_layout/schemas.py`
- Test: `tests/checkpoint/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CheckpointUsage(model_usage: dict[str, ModelUsage], role_usage: dict[str, ModelUsage], token_limit_usage: ModelUsage, cost: float, turns: int, time: float, working_time: float)` and `Checkpoint.usage: CheckpointUsage | None`, both importable from `inspect_ai.util._checkpoint._layout.schemas`.

A module-level `from inspect_ai.model._model_output import ModelUsage` in this file is safe — verified against every plausible first-import order. Do not add a lazy-import workaround.

- [ ] **Step 1: Write the failing tests**

Append to `tests/checkpoint/test_schemas.py`:

```python
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
        token_limit_usage=ModelUsage(
            input_tokens=10, output_tokens=5, total_tokens=15
        ),
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/checkpoint/test_schemas.py -k usage -v`
Expected: FAIL — `ImportError: cannot import name 'CheckpointUsage'`.

- [ ] **Step 3: Add the model**

In `src/inspect_ai/util/_checkpoint/_layout/schemas.py`, add the import next to the existing ones:

```python
from inspect_ai.model._model_output import ModelUsage
```

Add above `class Checkpoint`:

```python
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
```

Add to `Checkpoint`, after the `turn` field:

```python
    usage: CheckpointUsage | None = None
    """Sample usage as of this checkpoint. ``None`` for checkpoints written
    before this field existed, or fired with no live sample scope."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_schemas.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Verify no import cycle**

Run: `.venv/bin/python -c "import inspect_ai.event._checkpoint"`
Then: `.venv/bin/python -c "import inspect_ai.util._checkpoint._layout.schemas"`
Then: `.venv/bin/python -c "import inspect_ai.model._model_output"`
Expected: all three silent (exit 0).

- [ ] **Step 6: Commit**

```bash
git add src/inspect_ai/util/_checkpoint/_layout/schemas.py tests/checkpoint/test_schemas.py
git commit -m "Add CheckpointUsage to the checkpoint file schema"
```

---

### Task 2: `restore_usage` config flag

**Files:**
- Modify: `src/inspect_ai/util/_checkpoint/config.py`
- Modify: `src/inspect_ai/util/_checkpoint/parse_cli.py`
- Test: `tests/checkpoint/test_resolve.py`, `tests/checkpoint/test_parse.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CheckpointSampleConfig.restore_usage: bool | None` (inherited by `CheckpointConfig`) and `ResolvedCheckpointConfig.restore_usage: bool`.

The field goes on `CheckpointSampleConfig`, not `CheckpointConfig` — it is per-sample behaviour like `max_consecutive_failures`, not an eval-wide storage concern like `retention`.

- [ ] **Step 1: Write the failing merge tests**

Append to `tests/checkpoint/test_resolve.py`:

```python
def test_restore_usage_defaults_to_false() -> None:
    resolved = merge_checkpoint_configs(CheckpointConfig())
    assert resolved is not None
    assert resolved.restore_usage is False


def test_restore_usage_from_task_layer() -> None:
    resolved = merge_checkpoint_configs(_cfg("restore_usage", True))
    assert resolved is not None
    assert resolved.restore_usage is True


def test_restore_usage_sample_beats_task() -> None:
    resolved = merge_checkpoint_configs(
        _cfg("restore_usage", False), _sample_cfg("restore_usage", True)
    )
    assert resolved is not None
    assert resolved.restore_usage is True


def test_restore_usage_eval_beats_sample() -> None:
    resolved = merge_checkpoint_configs(
        None, _sample_cfg("restore_usage", True), _cfg("restore_usage", False)
    )
    assert resolved is not None
    assert resolved.restore_usage is False


def test_restore_usage_explicit_false_is_not_unset() -> None:
    resolved = merge_checkpoint_configs(
        _cfg("restore_usage", True), _sample_cfg("restore_usage", False)
    )
    assert resolved is not None
    assert resolved.restore_usage is False
```

The last test is the one that catches the likely bug: a merge written as `if layer.restore_usage:` instead of `if layer.restore_usage is not None:` would let a higher layer's explicit `False` fall through to the lower layer's `True`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/checkpoint/test_resolve.py -k restore_usage -v`
Expected: FAIL — `AttributeError: 'ResolvedCheckpointConfig' object has no attribute 'restore_usage'`.

- [ ] **Step 3: Add the field and merge it**

In `config.py`, add to `CheckpointSampleConfig` after `max_consecutive_failures`:

```python
    restore_usage: bool | None = None
    """Continue the sample's usage counters (tokens, cost, turns, time,
    working time) from its checkpoint on resume rather than restarting
    them at zero. ``None`` = inherit; defaults to ``False``."""
```

Add to `ResolvedCheckpointConfig` after `max_consecutive_failures`:

```python
    restore_usage: bool = False
```

In `merge_checkpoint_configs`, add to the declarations before the `for layer in (task, sample, eval_)` loop:

```python
    restore_usage: bool | None = None
```

Add inside that loop, after the `max_consecutive_failures` branch:

```python
        if layer.restore_usage is not None:
            restore_usage = layer.restore_usage
```

Add to the `ResolvedCheckpointConfig(...)` return:

```python
        restore_usage=restore_usage if restore_usage is not None else False,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_resolve.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing YAML test**

Append to `tests/checkpoint/test_parse.py`:

```python
def test_restore_usage_from_config_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"trigger": {"type": "turn", "every": 3}, "restore_usage": True})
    )

    cfg = _parse(str(path))

    assert cfg.restore_usage is True


def test_restore_usage_defaults_to_none_in_config_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"trigger": {"type": "turn", "every": 3}}))

    cfg = _parse(str(path))

    assert cfg.restore_usage is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/checkpoint/test_parse.py -k restore_usage -v`
Expected: FAIL — pydantic `ValidationError` for the extra key `restore_usage` (the model is `extra="forbid"`).

- [ ] **Step 7: Add it to the YAML model**

In `parse_cli.py`, add to `_CheckpointConfigModel` after `retention`:

```python
    restore_usage: bool | None = None
```

Add to `to_dataclass()`:

```python
            restore_usage=self.restore_usage,
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_parse.py tests/checkpoint/test_resolve.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/inspect_ai/util/_checkpoint/config.py \
        src/inspect_ai/util/_checkpoint/parse_cli.py \
        tests/checkpoint/test_resolve.py tests/checkpoint/test_parse.py
git commit -m "Add restore_usage checkpoint config flag (defaults off)"
```

---

### Task 3: Seedable limit nodes

**Files:**
- Modify: `src/inspect_ai/util/_limit.py`
- Test: `tests/util/test_limit_token.py`, `tests/util/test_limit_cost.py`, `tests/util/test_limit_turn.py`, `tests/util/test_limit_time.py`, `tests/util/test_limit_working.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `seed_limit_usage(*, token: Limit, cost: Limit, turn: Limit, time: Limit, working: Limit, token_usage: ModelUsage, cost_usage: float, turns: int, time_usage: float, working_usage: float) -> None`, importable from `inspect_ai.util._limit`. Not exported from `inspect_ai.util`.

Seeds must be applied **before** the node is entered — `_TimeLimit.__enter__` derives its cancel-scope deadline at entry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/util/test_limit_token.py`:

```python
def test_seeded_usage_accumulates_on_top() -> None:
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._limit import record_model_usage

    limit = token_limit(100)
    limit._seed_usage(ModelUsage(total_tokens=40))

    with limit:
        record_model_usage(ModelUsage(total_tokens=10))
        assert limit.usage == 50


def test_seeded_usage_counts_towards_the_ceiling() -> None:
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._limit import check_token_limit, record_model_usage

    limit = token_limit(100)
    limit._seed_usage(ModelUsage(total_tokens=95))

    with pytest.raises(LimitExceededError) as exc_info:
        with limit:
            record_model_usage(ModelUsage(total_tokens=10))
            check_token_limit()

    assert exc_info.value.value == 105
    assert exc_info.value.limit == 100
```

Append to `tests/util/test_limit_cost.py`:

```python
def test_seeded_cost_accumulates_on_top() -> None:
    from inspect_ai.util._limit import record_model_cost

    limit = cost_limit(1.0)
    limit._seed_usage(0.4)

    with limit:
        record_model_cost(0.1)
        assert limit.usage == pytest.approx(0.5)
```

Append to `tests/util/test_limit_turn.py`:

```python
def test_seeded_turns_accumulate_on_top() -> None:
    from inspect_ai.util._limit import record_turn

    limit = turn_limit(10)
    limit._seed_usage(4)

    with limit:
        record_turn()
        assert limit.usage == 5
```

Append to `tests/util/test_limit_time.py`:

```python
@pytest.mark.anyio
async def test_seeded_time_shortens_the_deadline() -> None:
    limit = time_limit(1.0)
    limit._seed_usage(0.9)

    with pytest.raises(LimitExceededError) as exc_info:
        with limit:
            await anyio.sleep(0.5)

    assert exc_info.value.limit == 1.0
    assert exc_info.value.value > 0.9


@pytest.mark.anyio
async def test_seeded_time_reports_cumulative_usage() -> None:
    limit = time_limit(10)
    limit._seed_usage(5.0)

    with limit:
        assert limit.usage >= 5.0
```

Append to `tests/util/test_limit_working.py`:

```python
@pytest.mark.anyio
async def test_seeded_working_time_accumulates_on_top() -> None:
    limit = working_limit(10)
    limit._seed_usage(6.0)

    with limit:
        assert limit.usage >= 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/util/test_limit_token.py tests/util/test_limit_cost.py tests/util/test_limit_turn.py tests/util/test_limit_time.py tests/util/test_limit_working.py -k seeded -v`
Expected: FAIL — `AttributeError: '_TokenLimit' object has no attribute '_seed_usage'` (and the equivalents).

- [ ] **Step 3: Add `_seed_usage` to each node**

In `_TokenLimit`, after `__init__`:

```python
    def _seed_usage(self, usage: ModelUsage) -> None:
        self._usage += usage
```

In `_CostLimit`:

```python
    def _seed_usage(self, cost: float) -> None:
        self._cost += cost
```

In `_TurnLimit`:

```python
    def _seed_usage(self, turns: int) -> None:
        self._turns += turns
```

In `_TimeLimit`, add `self._seeded_usage: float = 0.0` to `__init__`, then:

```python
    def _seed_usage(self, elapsed: float) -> None:
        self._seeded_usage += elapsed
```

Change `_TimeLimit.__enter__` to shorten the scope by the seed. Replace:

```python
        self._active_limit = self.limit
        self._cancel_scope = anyio.move_on_after(self._active_limit)
```

with:

```python
        self._active_limit = self.limit
        self._cancel_scope = anyio.move_on_after(self._remaining_limit())
```

and add:

```python
    def _remaining_limit(self) -> float | None:
        """The wall-clock budget left for this scope, after seeded usage.

        Clamped at 0 rather than going negative, which anyio reads as
        "already expired" — the intended outcome for a sample resuming
        with its time budget already spent.
        """
        if self._active_limit is None:
            return None
        return max(0.0, self._active_limit - self._seeded_usage)
```

Update `_refresh_deadline` so a live limit override re-derives against the seed. Replace:

```python
        self._cancel_scope.deadline = (
            self._start_time + self._active_limit
            if self._active_limit is not None
            else math.inf
        )
```

with:

```python
        remaining = self._remaining_limit()
        self._cancel_scope.deadline = (
            self._start_time + remaining if remaining is not None else math.inf
        )
```

Update `_TimeLimit.usage` to include the seed:

```python
    @property
    def usage(self) -> float:
        if self._start_time is None:
            return self._seeded_usage
        if self._end_time is None:
            return anyio.current_time() - self._start_time + self._seeded_usage
        return self._end_time - self._start_time + self._seeded_usage
```

Update the elapsed value reported by `_TimeLimit.__exit__`. Replace:

```python
            time_elapsed = self._end_time - self._start_time
```

with:

```python
            time_elapsed = self._end_time - self._start_time + self._seeded_usage
```

In `_WorkingLimit`, add `self._seeded_usage: float = 0.0` to `__init__`, then:

```python
    def _seed_usage(self, working: float) -> None:
        self._seeded_usage += working
```

and update its `usage` property:

```python
    @property
    def usage(self) -> float:
        if self._start_time is None:
            return self._seeded_usage
        if self._end_time is None:
            return (
                anyio.current_time()
                - self._start_time
                - self._waiting_time
                + self._seeded_usage
            )
        return (
            self._end_time - self._start_time - self._waiting_time + self._seeded_usage
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/util/ -k limit -v`
Expected: PASS, including all pre-existing limit tests.

- [ ] **Step 5: Write the failing test for the shared entry point**

Append to `tests/util/test_limit.py`:

```python
def test_seed_limit_usage_seeds_every_node() -> None:
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._limit import (
        cost_limit,
        seed_limit_usage,
        time_limit,
        token_limit,
        turn_limit,
        working_limit,
    )

    token = token_limit(100)
    cost = cost_limit(1.0)
    turn = turn_limit(10)
    time_ = time_limit(60)
    working = working_limit(60)

    seed_limit_usage(
        token=token,
        cost=cost,
        turn=turn,
        time=time_,
        working=working,
        token_usage=ModelUsage(total_tokens=40),
        cost_usage=0.4,
        turns=4,
        time_usage=12.0,
        working_usage=10.0,
    )

    assert token.usage == 40
    assert cost.usage == pytest.approx(0.4)
    assert turn.usage == 4
    assert time_.usage == pytest.approx(12.0)
    assert working.usage == pytest.approx(10.0)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/pytest tests/util/test_limit.py -k seed_limit_usage -v`
Expected: FAIL — `ImportError: cannot import name 'seed_limit_usage'`.

- [ ] **Step 7: Add the entry point**

Add to `_limit.py`, next to the other module-level helpers:

```python
def seed_limit_usage(
    *,
    token: Limit,
    cost: Limit,
    turn: Limit,
    time: Limit,
    working: Limit,
    token_usage: ModelUsage,
    cost_usage: float,
    turns: int,
    time_usage: float,
    working_usage: float,
) -> None:
    """Pre-load sample limit nodes with usage from a prior attempt.

    Internal: used by checkpoint resume so a continued sample enforces
    against its cumulative usage. Call before entering any of the nodes —
    a time limit derives its deadline at ``__enter__``.
    """
    if (
        not isinstance(token, _TokenLimit)
        or not isinstance(cost, _CostLimit)
        or not isinstance(turn, _TurnLimit)
        or not isinstance(time, _TimeLimit)
        or not isinstance(working, _WorkingLimit)
    ):
        raise TypeError("seed_limit_usage requires concrete sample limit nodes")
    token._seed_usage(token_usage)
    cost._seed_usage(cost_usage)
    turn._seed_usage(turns)
    time._seed_usage(time_usage)
    working._seed_usage(working_usage)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/util/ -k limit -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/inspect_ai/util/_limit.py tests/util/test_limit.py \
        tests/util/test_limit_token.py tests/util/test_limit_cost.py \
        tests/util/test_limit_turn.py tests/util/test_limit_time.py \
        tests/util/test_limit_working.py
git commit -m "Allow sample limit nodes to be seeded with prior usage"
```

---

### Task 4: Prior-usage offsets in `SampleTiming`

**Files:**
- Modify: `src/inspect_ai/_util/working.py`
- Test: `tests/util/test_working_time.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `init_sample_working_time(start_time: float, *, prior_time: float = 0.0, prior_working_time: float = 0.0) -> None` and a new `sample_total_time() -> float`, both in `inspect_ai._util.working`. `sample_working_time()` keeps its signature and now includes the offset.

`working_time` must be derived from *un-offset* elapsed, then have `prior_working_time` added. Deriving it as `total_time - waiting` (what `run.py` does today) would count `prior_time` twice.

- [ ] **Step 1: Write the failing test**

Create `tests/util/test_working_time.py`:

```python
"""Prior-usage offsets on the sample timing contextvar."""

from __future__ import annotations

import time

from inspect_ai._util.working import (
    init_sample_working_time,
    report_sample_waiting_time,
    sample_total_time,
    sample_working_time,
)


def test_defaults_to_no_offset() -> None:
    init_sample_working_time(time.monotonic())

    assert sample_total_time() < 1.0
    assert sample_working_time() < 1.0


def test_prior_time_offsets_total() -> None:
    init_sample_working_time(time.monotonic(), prior_time=100.0)

    assert 100.0 <= sample_total_time() < 101.0


def test_prior_working_time_offsets_working() -> None:
    init_sample_working_time(time.monotonic(), prior_working_time=50.0)

    assert 50.0 <= sample_working_time() < 51.0


def test_offsets_are_independent() -> None:
    init_sample_working_time(
        time.monotonic(), prior_time=100.0, prior_working_time=50.0
    )
    report_sample_waiting_time(10.0)

    assert 100.0 <= sample_total_time() < 101.0
    # working excludes the 10s wait and adds only its own offset
    assert 40.0 <= sample_working_time() < 41.0
```

`test_offsets_are_independent` is the regression guard for the double-count: if `working_time` were derived from the offset total, it would read ~140 rather than ~40.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/util/test_working_time.py -v`
Expected: FAIL — `ImportError: cannot import name 'sample_total_time'`.

- [ ] **Step 3: Implement the offsets**

In `src/inspect_ai/_util/working.py`, add to `SampleTiming`:

```python
    prior_time: float = 0.0
    prior_working_time: float = 0.0
```

Replace `init_sample_working_time`:

```python
def init_sample_working_time(
    start_time: float,
    *,
    prior_time: float = 0.0,
    prior_working_time: float = 0.0,
) -> None:
    _sample_timing.set(
        SampleTiming(
            start_time=start_time,
            start_datetime=datetime.now(timezone.utc),
            prior_time=prior_time,
            prior_working_time=prior_working_time,
        )
    )
```

Replace `sample_working_time` and add `sample_total_time`:

```python
def sample_total_time() -> float:
    timing = _sample_timing.get()
    return time.monotonic() - timing.start_time + timing.prior_time


def sample_working_time() -> float:
    timing = _sample_timing.get()
    return (
        time.monotonic()
        - timing.start_time
        - timing.waiting_time
        + timing.prior_working_time
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/util/test_working_time.py -v`
Expected: PASS.

- [ ] **Step 5: Check no existing caller regressed**

Run: `.venv/bin/pytest tests/util/ tests/test_sample_limits.py -q`
Expected: PASS. Existing callers pass no offsets, so every value is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/inspect_ai/_util/working.py tests/util/test_working_time.py
git commit -m "Support prior-attempt offsets on sample timing"
```

---

### Task 5: Capture usage at each fire

**Files:**
- Modify: `src/inspect_ai/util/_checkpoint/checkpointer_impl.py`
- Test: `tests/checkpoint/test_checkpointer.py`

**Interfaces:**
- Consumes: `CheckpointUsage` (Task 1).
- Produces: `Checkpoint.usage` populated on every fire that has a live sample scope.

`sample_limits()` raises `RuntimeError` outside a sample scope, and `tests/checkpoint/test_checkpointer.py` drives real fires with only `sample_state()` and `transcript()` patched. Capture must return `None` in that case, not raise.

- [ ] **Step 1: Write the failing test**

Append to `tests/checkpoint/test_checkpointer.py`:

```python
async def test_fire_without_limit_scope_records_no_usage(tmp_path: Path) -> None:
    from inspect_ai.util._checkpoint.checkpointer_impl import _capture_usage

    assert _capture_usage() is None
```

Adjust the import path and any fixture use to match the file's existing conventions; the assertion is the point.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/checkpoint/test_checkpointer.py -k without_limit_scope -v`
Expected: FAIL — `ImportError: cannot import name '_capture_usage'`.

- [ ] **Step 3: Implement capture**

Add to `checkpointer_impl.py`:

```python
def _capture_usage() -> CheckpointUsage | None:
    """Sample usage right now, or ``None`` outside a sample limit scope.

    Fires driven outside a sample (tests, direct harness use) have no
    limit trees, and ``sample_limits()`` raises rather than returning
    zeros.
    """
    from inspect_ai.model._model import sample_model_usage, sample_role_usage
    from inspect_ai.util._limit import sample_limits

    try:
        limits = sample_limits()
    except RuntimeError:
        return None

    return CheckpointUsage(
        model_usage=deepcopy(sample_model_usage()),
        role_usage=deepcopy(sample_role_usage()),
        token_limit_usage=deepcopy(limits.token._usage),
        cost=limits.cost.usage,
        turns=int(limits.turn.usage),
        time=limits.time.usage,
        working_time=limits.working.usage,
    )
```

Add `from copy import deepcopy` and `from ._layout.schemas import Checkpoint, CheckpointUsage` to the imports (extend the existing `schemas` import rather than adding a second one).

`limits.token._usage` is the raw `ModelUsage` accumulator; `limits.token.usage` is the *metered* float, which loses the per-field breakdown a resume needs to re-meter.

In `_fire_once`, add to the `Checkpoint(...)` construction after `turn=self._turn,`:

```python
                    usage=_capture_usage(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_checkpointer.py -v`
Expected: PASS, including all pre-existing fire-driving tests.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_ai/util/_checkpoint/checkpointer_impl.py tests/checkpoint/test_checkpointer.py
git commit -m "Record sample usage in each checkpoint"
```

---

### Task 6: Carry usage onto `ResumeCheckpoint`

**Files:**
- Modify: `src/inspect_ai/util/_checkpoint/checkpointer.py`
- Modify: `src/inspect_ai/_eval/task/run.py:2690-2720`
- Test: `tests/checkpoint/test_sample_checkpoints.py`

**Interfaces:**
- Consumes: `CheckpointUsage` (Task 1).
- Produces: `ResumeCheckpoint.usage: CheckpointUsage | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/checkpoint/test_sample_checkpoints.py`:

```python
async def test_resume_carries_usage(tmp_path: Path) -> None:
    from inspect_ai._eval.task.run import _resume_if_checkpointed
    from inspect_ai.util._checkpoint._layout.schemas import (
        Checkpoint,
        CheckpointUsage,
        SnapshotDetails,
    )
    from inspect_ai.util._checkpoint._layout.sample_checkpoints_dir import (
        write_checkpoint_file,
    )

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
```

Match the file's existing import block and fixtures — `sample_checkpoints_dir` and the datetime imports may already be present.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/checkpoint/test_sample_checkpoints.py -k carries_usage -v`
Expected: FAIL — `AttributeError: 'ResumeCheckpoint' object has no attribute 'usage'`.

- [ ] **Step 3: Add the field and populate it**

In `checkpointer.py`, add to `ResumeCheckpoint`:

```python
    usage: CheckpointUsage | None = None
```

Add the import:

```python
from ._layout.schemas import CheckpointUsage
```

In `run.py`, at the `ResumeCheckpoint(...)` construction around line 2713, add:

```python
        usage=checkpoint.usage if checkpoint is not None else None,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_sample_checkpoints.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inspect_ai/util/_checkpoint/checkpointer.py \
        src/inspect_ai/_eval/task/run.py \
        tests/checkpoint/test_sample_checkpoints.py
git commit -m "Carry checkpoint usage onto ResumeCheckpoint"
```

---

### Task 7: Apply the seeds when opted in

**Files:**
- Modify: `src/inspect_ai/model/_model.py:2597`
- Modify: `src/inspect_ai/_eval/task/run.py:1718`, `:1953`, `:1963-1977`, `:2613`, `:2645`
- Test: `tests/checkpoint/test_sample_checkpoints.py`

**Interfaces:**
- Consumes: `ResolvedCheckpointConfig.restore_usage` (Task 2), `seed_limit_usage` (Task 3), `init_sample_working_time(..., prior_time=, prior_working_time=)` and `sample_total_time()` (Task 4), `ResumeCheckpoint.usage` (Task 6).
- Produces: the observable behaviour the e2e tests in Task 8 assert.

This is the task where all the pieces meet. Both inputs are available at the seeding point: `resolved_checkpoint` is built at `run.py:1758`, `resume_checkpoint` is a parameter, and the limit scopes do not open until `run.py:1963`.

**Two things not to change:**

- **Do not touch `scoring_time_limit` at `run.py:2205`.** It stays `time_limit / 2` of the *configured* limit. It looks like it should shrink by prior usage; it must not. Its purpose is to give scoring a fair shot precisely when the agent's clock ran out — usually on a hung container — so shrinking it by prior usage would leave a twice-crashed sample a window too short to reach that container.
- **Do not special-case `attempt == "resume_for_scoring"`.** `_prior_usage` deliberately does not discriminate on attempt type. Scoring has no token, cost, turn, message, or working limit in scope (they exit before `run.py:2205`), so seeding cannot affect enforcement there — it only makes the log totals cumulative, which is wanted on every attempt type.

- [ ] **Step 1: Make sample model usage seedable**

In `src/inspect_ai/model/_model.py`, replace `init_sample_model_data`:

```python
def init_sample_model_data(
    model_usage: dict[str, ModelUsage] | None = None,
    role_usage: dict[str, ModelUsage] | None = None,
) -> None:
    """Initialize all per-sample model accumulators (usage, role usage, fallbacks)."""
    sample_model_usage_context_var.set(deepcopy(model_usage) if model_usage else {})
    sample_role_usage_context_var.set(deepcopy(role_usage) if role_usage else {})
    init_sample_model_fallbacks()
```

Add `from copy import deepcopy` if it is not already imported. Leave `init_sample_model_usage()` and `init_sample_role_usage()` in place — other callers use them.

- [ ] **Step 2: Run the model tests to check nothing regressed**

Run: `.venv/bin/pytest tests/model/ -q -x`
Expected: PASS. The no-argument call sites behave identically.

- [ ] **Step 3: Write the failing integration test**

Append to `tests/checkpoint/test_sample_checkpoints.py`:

```python
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
    from inspect_ai.util._checkpoint.checkpointer import ResumeCheckpoint
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig
    from inspect_ai.util._checkpoint._triggers import Manual

    resume = ResumeCheckpoint(
        sample_checkpoints_dir="/tmp/x", attempt="resume", usage=_usage_seed()
    )
    config = ResolvedCheckpointConfig(trigger=Manual(), restore_usage=True)

    assert _prior_usage(resume, config) is not None


def test_prior_usage_is_none_when_flag_off() -> None:
    from inspect_ai._eval.task.run import _prior_usage
    from inspect_ai.util._checkpoint.checkpointer import ResumeCheckpoint
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig
    from inspect_ai.util._checkpoint._triggers import Manual

    resume = ResumeCheckpoint(
        sample_checkpoints_dir="/tmp/x", attempt="resume", usage=_usage_seed()
    )
    config = ResolvedCheckpointConfig(trigger=Manual())

    assert _prior_usage(resume, config) is None


def test_prior_usage_is_none_without_a_resume() -> None:
    from inspect_ai._eval.task.run import _prior_usage
    from inspect_ai.util._checkpoint.config import ResolvedCheckpointConfig
    from inspect_ai.util._checkpoint._triggers import Manual

    config = ResolvedCheckpointConfig(trigger=Manual(), restore_usage=True)

    assert _prior_usage(None, config) is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/checkpoint/test_sample_checkpoints.py -k prior_usage -v`
Expected: FAIL — `ImportError: cannot import name '_prior_usage'`.

- [ ] **Step 5: Add the gate helper**

In `run.py`, next to `_resume_if_checkpointed`:

```python
def _prior_usage(
    resume_checkpoint: ResumeCheckpoint | None,
    config: ResolvedCheckpointConfig | None,
) -> CheckpointUsage | None:
    """The usage a resumed sample should continue from, if it should.

    ``None`` unless this is a resume of a checkpoint that recorded usage
    and the eval opted in via ``restore_usage``.
    """
    if resume_checkpoint is None or config is None or not config.restore_usage:
        return None
    return resume_checkpoint.usage
```

Imports needed in `run.py`: add `CheckpointUsage` to the existing `_layout.schemas` import and `ResolvedCheckpointConfig` to the existing `_checkpoint.config` import; add `Limit` and `seed_limit_usage` to the `from inspect_ai.util._limit import (...)` block at line 160. `LimitExceededError` is already imported there (line 161).

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/checkpoint/test_sample_checkpoints.py -k prior_usage -v`
Expected: PASS.

- [ ] **Step 7: Wire the seeds into the sample run**

The merge currently sits at `run.py:1758`, *after* `init_sample_model_data()` at `run.py:1718`. Hoist it: the merge reads only `checkpoint`, `sample.checkpoint`, `eval_checkpoint`, and the two `task` hooks, and `sample` exists from `run.py:1694`. Move the whole `resolved_checkpoint = merge_checkpoint_configs(...)` block (and its comment) to immediately after `reset_sample_limit_data()` at `run.py:1699`, and append:

```python
        prior_usage = _prior_usage(resume_checkpoint, resolved_checkpoint)
```

Then seed the existing call at `run.py:1718` in place:

```python
        init_sample_model_data(
            prior_usage.model_usage if prior_usage else None,
            prior_usage.role_usage if prior_usage else None,
        )
```

Do not leave a second `merge_checkpoint_configs` call behind at 1758 — later code reads the same `resolved_checkpoint` local.

At `run.py:1953`, seed the timing contextvar:

```python
                        init_sample_working_time(
                            start_time,
                            prior_time=prior_usage.time if prior_usage else 0.0,
                            prior_working_time=(
                                prior_usage.working_time if prior_usage else 0.0
                            ),
                        )
```

At `run.py:1963`, seed the limit nodes between construction and entry:

```python
                        sample_time_limit = create_time_limit(time_limit)
                        sample_turn_limit = create_turn_limit(turn_limit)
                        sample_working_limit = create_working_limit(working_limit)
                        if prior_usage is not None:
                            seed_limit_usage(
                                token=state._token_limit,
                                cost=state._cost_limit,
                                turn=sample_turn_limit,
                                time=sample_time_limit,
                                working=sample_working_limit,
                                token_usage=prior_usage.token_limit_usage,
                                cost_usage=prior_usage.cost,
                                turns=prior_usage.turns,
                                time_usage=prior_usage.time,
                                working_usage=prior_usage.working_time,
                            )
                        with (
```

and replace the inline `create_turn_limit(turn_limit)` / `create_working_limit(working_limit)` entries in the `with (...)` list with `sample_turn_limit` / `sample_working_limit`. Leave the rest of the list untouched.

- [ ] **Step 8: Make the sample log cumulative**

In `create_eval_sample` (`run.py:2613`), replace:

```python
    total_time = time.monotonic() - start_time if start_time is not None else None
```

with:

```python
    total_time = sample_total_time() if start_time is not None else None
```

and replace the `working_time` argument (`run.py:2645`):

```python
        working_time=round(sample_working_time(), 3)
        if total_time is not None
        else None,
```

Import `sample_total_time` and `sample_working_time` from `inspect_ai._util.working` alongside the existing `sample_waiting_time` import.

`working_time` must come from `sample_working_time()`, not `total_time - sample_waiting_time()` — the latter would add `prior_time` into a working-time figure that should only carry `prior_working_time`.

- [ ] **Step 9: Guard against a seed that already exceeds its limit**

Add immediately before the `with (` at `run.py:1963`, after the seeding block:

```python
                        if prior_usage is not None:
                            _raise_if_prior_usage_exhausted(
                                token=state._token_limit,
                                cost=state._cost_limit,
                                turn=sample_turn_limit,
                                time=sample_time_limit,
                                working=sample_working_limit,
                            )
```

and define near `_prior_usage`:

```python
def _raise_if_prior_usage_exhausted(
    *, token: Limit, cost: Limit, turn: Limit, time: Limit, working: Limit
) -> None:
    """Fail a resume whose seeded usage already meets its limit.

    Reachable when a limit was lowered between attempts. Raising here
    rather than letting the scopes open keeps a zero-budget time limit
    from cancelling the sample partway through sandbox restore.
    """
    for limit_type, node in (
        ("token", token),
        ("cost", cost),
        ("turns", turn),
        ("time", time),
        ("working", working),
    ):
        ceiling = node.limit
        if ceiling is not None and node.usage >= ceiling:
            raise LimitExceededError(
                limit_type,
                value=node.usage,
                limit=ceiling,
                message=(
                    f"Restored usage from checkpoint already meets the "
                    f"{limit_type} limit. usage: {node.usage}, limit: {ceiling}"
                ),
                source=node,
            )
```

`LimitExceededError`'s `type` parameter accepts the literals `"token"`, `"cost"`, `"turns"`, `"time"`, `"working"` — confirm against its definition at `_limit.py:40` and adjust the strings if they differ.

- [ ] **Step 10: Write the failing test for the guard**

Append to `tests/util/test_limit.py`:

```python
def test_seeded_node_at_its_ceiling_is_detectable() -> None:
    from inspect_ai.model._model_output import ModelUsage
    from inspect_ai.util._limit import token_limit

    limit = token_limit(100)
    limit._seed_usage(ModelUsage(total_tokens=100))

    assert limit.usage >= limit.limit
```

- [ ] **Step 11: Run the full affected suite**

Run: `.venv/bin/pytest tests/checkpoint/ tests/util/ tests/test_sample_limits.py -q`
Expected: PASS.

Run: `.venv/bin/pytest tests/test_eval.py -q`
Expected: PASS — nothing opts in, so every sample behaves as before.

- [ ] **Step 12: Type check and commit**

```bash
.venv/bin/ruff format && .venv/bin/ruff check --fix
.venv/bin/mypy --exclude tests/test_package src tests
git add src/inspect_ai/model/_model.py src/inspect_ai/_eval/task/run.py \
        tests/checkpoint/test_sample_checkpoints.py tests/util/test_limit.py
git commit -m "Continue a resumed sample's usage when restore_usage is set"
```

---

### Task 8: End-to-end coverage and docs

**Files:**
- Modify: `tests/checkpoint/resume_kill_harness.py`
- Modify: `tests/checkpoint/test_checkpoint_e2e.py`
- Modify: `docs/checkpointing.qmd`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: no code interfaces.

Requires Docker. The harness kills a child process with a real `SIGKILL` and uses a scripted mock model, so token counts are exact rather than approximate.

- [ ] **Step 1: Parametrize the harness on the flag**

In `resume_kill_harness.py`, read both knobs from the environment so the parent test can vary them per child process. Add next to the file's existing `*_ENV` constants:

```python
RESTORE_USAGE_ENV = "INSPECT_TEST_RESTORE_USAGE"
TOKEN_LIMIT_ENV = "INSPECT_TEST_TOKEN_LIMIT"
```

In the `Task(...)` the harness builds:

```python
        checkpoint=CheckpointConfig(
            trigger=TurnInterval(every=1),
            restore_usage=os.environ.get(RESTORE_USAGE_ENV) == "1",
        ),
        token_limit=(
            int(limit) if (limit := os.environ.get(TOKEN_LIMIT_ENV)) else None
        ),
```

`token_limit` is a `Task` field, not a `CheckpointConfig` one — check its exact spelling against `Task.__init__` before wiring it.

- [ ] **Step 2: Write the failing e2e test**

Append to `test_checkpoint_e2e.py`. The helper mirrors the kill → resume → run-to-completion sequence in `test_checkpoint_resume_rehydrated_event_layout` (line 312), reusing its existing module-level helpers:

```python
def _kill_then_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    restore_usage: bool,
) -> EvalLog:
    """One SIGKILL'd attempt, then a resume that runs to completion."""
    cancel_file = tmp_path / f"cancels-{int(restore_usage)}.txt"
    monkeypatch.setenv(CANCEL_FILE_ENV, str(cancel_file))
    monkeypatch.setenv(TARGET_ENV, "2")
    monkeypatch.setenv(RESTORE_USAGE_ENV, "1" if restore_usage else "0")
    cancel_file.unlink(missing_ok=True)

    log_dir = str(tmp_path / f"logs-{int(restore_usage)}")
    tests_dir = Path(__file__).parent.parent

    projects_before = _inspect_projects()
    try:
        _run_interrupted_attempt(log_dir, None, tests_dir, "SIGKILL")
        reset_generates()
        return eval_retry(read_eval_log(_latest_log(log_dir)), log_dir=log_dir)[0]
    finally:
        for name in _inspect_projects() - projects_before:
            _force_remove_project(name)


def _sample_tokens(log: EvalLog) -> int:
    assert log.samples is not None and len(log.samples) == 1
    return sum(u.total_tokens for u in log.samples[0].model_usage.values())


@pytest.mark.slow
@skip_if_no_docker
@flaky_retry
def test_restore_usage_carries_prior_attempt_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag on, the resumed sample's log spans both attempts.

    Compared against the same run with the flag off rather than a
    hardcoded token count — the scripted model makes both runs
    deterministic, so the difference is exactly the restored usage.
    """
    off = _kill_then_resume(tmp_path, monkeypatch, restore_usage=False)
    on = _kill_then_resume(tmp_path, monkeypatch, restore_usage=True)

    assert off.status == "success" and on.status == "success"
    assert _sample_tokens(on) > _sample_tokens(off)

    on_sample = on.samples[0] if on.samples else None
    off_sample = off.samples[0] if off.samples else None
    assert on_sample is not None and off_sample is not None
    assert on_sample.total_time is not None and off_sample.total_time is not None
    assert on_sample.total_time > off_sample.total_time
    assert on_sample.working_time is not None
    assert on_sample.working_time <= on_sample.total_time
```

Add `RESTORE_USAGE_ENV` to the existing `from checkpoint.resume_kill_harness import (...)` block, and `EvalLog` to the `inspect_ai.log` imports.

Running both arms in one test keeps the comparison honest — nothing is hardcoded, and the `off` arm doubles as the regression guard that the default is unchanged.

- [ ] **Step 2b: Assert the flag actually gates enforcement**

Also append a test that a limit sized to span attempts trips only when opted in. Reuse `_kill_then_resume` with a token limit applied through the harness (add a `TOKEN_LIMIT_ENV` read in `resume_kill_harness.py`'s `Task(...)` construction, mirroring how `RESTORE_USAGE_ENV` was added in Step 1):

```python
@pytest.mark.slow
@skip_if_no_docker
@flaky_retry
def test_restore_usage_enforces_the_spanning_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget that two attempts collectively exceed stops only the one with
    restore_usage=True."""
    off = _kill_then_resume(tmp_path, monkeypatch, restore_usage=False)
    budget = _sample_tokens(off) + 1
    monkeypatch.setenv(TOKEN_LIMIT_ENV, str(budget))
    on = _kill_then_resume(tmp_path, monkeypatch, restore_usage=True)

    assert off.samples is not None and off.samples[0].limit is None
    assert on.samples is not None and on.samples[0].limit is not None
    assert on.samples[0].limit.type == "token"
```

Setting the budget from the `off` arm's measured usage means the test does not need to know the model's token counts at all.

- [ ] **Step 3: Run the e2e tests**

Run: `.venv/bin/pytest tests/checkpoint/test_checkpoint_e2e.py -k restore_usage -v`
Expected: PASS (both). These are slow and need Docker. If `test_restore_usage_carries_prior_attempt_usage` shows the two arms equal, the seeding never took effect; if it shows the `off` arm already cumulative, the seeding is leaking into evals that did not opt in. Fix either before continuing.

- [ ] **Step 4: Update the feature docs**

In `docs/checkpointing.qmd`, add a fourth item to the numbered list in the "Recovery" section:

```markdown
4.  Usage (opt-in): with `restore_usage=True`, the sample's token, cost, turn, time, and working-time counters continue from the checkpoint instead of restarting at zero. Off by default. The message limit always continues, since it counts the restored conversation.
```

Add a row to the `CheckpointConfig` table under "Configuration":

```markdown
| `restore_usage` | Continue the sample's usage counters from its checkpoint on resume rather than restarting them at zero. Defaults to `False`. |
```

Add after that table:

```markdown
Whether to enable `restore_usage` is a trade-off. Left off, a sample that
crashes repeatedly can spend a multiple of its configured budget — each
attempt gets the full allowance. Turned on, a sample can resume with almost
nothing left and fail immediately, turning a recoverable crash into a lost
sample. Off is the default because checkpointing's primary job is
resilience.

Usage is restored as of the checkpoint, not as of the crash: work done after
the last checkpoint is rolled back, so it is not charged. Time spent
restoring *is* charged to the time and working limits.
```

Add to "Limitations":

```markdown
- Restore is not free — Restic restore, sandbox ingress, and the task's `on_resume` hook all run inside the sample's time and working limits.
```

- [ ] **Step 5: Add the CHANGELOG entry**

Add as the first item under `## Unreleased` in `CHANGELOG.md`:

```markdown
- Checkpointing: New `restore_usage` option continues a resumed sample's token, cost, turn, time, and working-time usage from its checkpoint instead of restarting at zero (off by default).
```

- [ ] **Step 6: Verify the entry's placement**

Run: `git diff "$(git merge-base origin/main HEAD)" HEAD -- CHANGELOG.md`
Expected: the added line sits under `## Unreleased`. Note `origin/main` on this repo is the METR fork and lags local `main`; if the merge-base is unhelpful, check placement by eye instead.

- [ ] **Step 7: Commit**

```bash
git add tests/checkpoint/resume_kill_harness.py \
        tests/checkpoint/test_checkpoint_e2e.py \
        docs/checkpointing.qmd CHANGELOG.md
git commit -m "Cover checkpoint usage restoration end-to-end and document it"
```

---

### Task 9: Regenerate the log schema and TypeScript types

**Files:**
- Modify: `src/inspect_ai/_view/inspect-openapi.json`
- Modify: `src/inspect_ai/_view/ts-mono` (submodule)

**Interfaces:**
- Consumes: `CheckpointUsage` (Task 1).
- Produces: nothing consumed by other tasks.

`CheckpointEvent` inherits from `Checkpoint`, so `usage` lands on the event and therefore in the log schema. `check-schema-and-types` fails until both artifacts are regenerated. Expect roughly a 55-line addition to `inspect-openapi.json`: a `CheckpointUsage` component and a `usage` property on `CheckpointEvent`.

- [ ] **Step 1: Confirm the submodule is checked out at the recorded pointer**

Run: `git submodule status src/inspect_ai/_view/ts-mono`
Expected: the SHA with **no** leading `+` or `-`. A `-` means it is not initialized: run `git submodule update --init --recursive src/inspect_ai/_view/ts-mono`. A `+` means it is at the wrong commit: reset it to the pointer recorded at HEAD before doing anything else.

- [ ] **Step 2: Regenerate the OpenAPI schema**

Run: `.venv/bin/python src/inspect_ai/_view/schema.py`
Then: `git diff --stat -- src/inspect_ai/_view/inspect-openapi.json`
Expected: a non-empty diff adding `CheckpointUsage` and a `usage` property.

- [ ] **Step 3: Regenerate the TypeScript types**

Run: `pnpm install --frozen-lockfile` in `src/inspect_ai/_view/ts-mono`, then `pnpm --filter @tsmono/inspect-common types:generate`.
Then: `git -C src/inspect_ai/_view/ts-mono status --short`
Expected: `packages/inspect-common/src/types/generated.ts` modified.

- [ ] **Step 4: Land via the land-ts-mono skill**

Read `.claude/skills/land-ts-mono/SKILL.md` and follow it. Do **not** hand-commit the submodule gitlink — that skill owns the ordering between the ts-mono commit and the gitlink bump, and getting it wrong produces a repo that does not build from a fresh clone.

- [ ] **Step 5: Verify the sync check would pass**

Run: `pnpm --filter @tsmono/inspect-common types:generate` in `src/inspect_ai/_view/ts-mono`, then `git -C src/inspect_ai/_view/ts-mono diff --exit-code -- packages/inspect-common/src/types/generated.ts`
Expected: exit 0 — regenerating a second time produces no diff.

---

## Final verification

- [ ] `.venv/bin/ruff format --check && .venv/bin/ruff check`
- [ ] `.venv/bin/mypy --exclude tests/test_package src tests`
- [ ] `.venv/bin/pytest tests/checkpoint/ tests/util/ tests/test_sample_limits.py -q`
- [ ] `.venv/bin/pytest tests/test_eval.py -q`
- [ ] `git status` shows no unintended submodule modification
- [ ] The CHANGELOG entry is under `## Unreleased`
