# Carrying usage limits across a checkpoint resume

> **Status: design.** Companion to [`docs/checkpointing.qmd`](../docs/checkpointing.qmd) (the user-facing feature docs) and [`recover.md`](recover.md). Owns the semantics of what a resumed sample's limits and usage totals mean.

## The problem

Checkpointing restores a sample's messages, sandbox filesystem, store, and event history — but not a single one of its usage counters. A sample that has burned 400K tokens and 50 minutes, then crashes and resumes, restarts with zero of both.

Every counter resets, and each for its own reason:

| Counter | Why it resets |
|---|---|
| Tokens | `_TokenLimit` is constructed per attempt (`run.py:1963-1976`); `init_sample_model_data()` (`run.py:1718`) clears `sample_model_usage_context_var` |
| Cost | `_CostLimit` constructed per attempt |
| Turns | `_TurnLimit` constructed per attempt |
| Time | `_TimeLimit.__enter__` bakes an `anyio.move_on_after` deadline from *now* |
| Working time | `init_sample_working_time(start_time)` (`run.py:1953`) restarts the clock |

The one counter that *does* survive is the **message limit** — it is checked against `len(state.messages)`, and messages are restored from `agent_state.json`. It needs no work, and its behaviour is the target the others should match.

The consequences are twofold. A `--token-limit`-bounded sample that crashes twice can spend three times its configured budget. And the resumed sample's log under-reports what it actually cost: `EvalSample.model_usage`, `total_time`, and `working_time` all describe the final attempt only, even though the restored transcript replays the prior attempt's model events under a `prior_run` span.

## Scope

**In scope.** Persist a sample's usage at each checkpoint fire and seed it back on resume, so tokens, cost, turns, time, and working time all continue rather than restart — and so the log reports the sample's cumulative cost.

**Explicitly out of scope: excluding restore time from the clocks.** Restore work (restic restore, sandbox ingress, the task's `on_resume` hook) runs inside the sample's time and working limits and will continue to. So does fresh-sample checkpointer setup and each fire's backup cost. This was considered and deliberately dropped: it needs a separate mechanism (pushing out a live `anyio` cancel-scope deadline mid-flight, the same problem `pause-resume.md` names as the reason a paused sample keeps burning its `time_limit`), and it is a different question from the one this design answers. Everything below is achievable with construction-time seeding alone.

## Design

### 1. What is persisted

A new `usage` block on `Checkpoint` in `_layout/schemas.py`, written into `ckpt-NNNNN.json` at every fire:

```python
class CheckpointUsage(BaseModel):
    """Sample usage as of this checkpoint, for continuation on resume."""

    model_config = ConfigDict(extra="allow")

    model_usage: dict[str, ModelUsage] = Field(default_factory=dict)
    """Per-model usage — `sample_model_usage()`."""

    role_usage: dict[str, ModelUsage] = Field(default_factory=dict)
    """Per-role usage — `sample_role_usage()`."""

    token_limit_usage: ModelUsage = Field(default_factory=ModelUsage)
    """The token limit node's own accumulator. Not derivable from
    `model_usage` — see below."""

    cost: float = 0.0
    turns: int = 0
    time: float = 0.0
    working_time: float = 0.0


class Checkpoint(BaseModel):
    ...
    usage: CheckpointUsage | None = None
    """Sample usage as of this checkpoint. `None` when the checkpoint was
    written before this field existed, or fired with no live sample scope
    (see §5)."""
```

Captured in `_fire_once()` (`checkpointer_impl.py:544-560`) alongside the existing `turn` / `duration_ms` fields, reading `sample_limits()` and the model-usage contextvars.

**Why the checkpoint metadata file and not a host-context file.** `ckpt-NNNNN.json` is parsed by `scan_latest_committed_checkpoint()` from `_resume_if_checkpointed()` (`run.py:2690`) *before the sample runs*. Host-context files (`store.json`, `agent_state.json`, …) only materialize after restic restore, deep inside `hydrate()` — which the agent triggers from `async with checkpointer()`, long after the limit nodes are constructed and their clocks started. Persisting in the metadata file is what allows every value to be applied as a construction-time seed rather than a mid-flight mutation of running limits.

**Why both `model_usage` and `token_limit_usage`.** They are not the same number. `suspend_token_limit()` lets tokens land in sample accounting while the limit deliberately ignores them, so the two diverge by exactly the suspended tokens. Deriving the limit's accumulator from `model_usage` would silently charge suspended tokens to the limit on resume. The same reasoning applies to `turns`, which is read from the turn limit node rather than the checkpointer's own `self._turn` tick counter — `suspend_turn_limit()` exists for the same purpose.

**Not persisted: message count.** The message limit already carries correctly via restored messages. Persisting it would create a second source of truth that could disagree with `len(state.messages)`.

**Backward compatibility.** `Checkpoint` already declares `extra="allow"`, and every field above has a zero default, so a checkpoint written before this change resumes with zero prior usage — today's behaviour exactly. No migration, no version gate.

### 2. How it is restored

`ResumeCheckpoint` (`checkpointer.py:34`) gains `usage: CheckpointUsage | None`, populated at its single construction site (`run.py:2713`) from the `Checkpoint` that `_resume_if_checkpointed()` has already parsed.

In `task_run_sample`, all applied before the limit scopes open at `run.py:1963`:

| Value | Destination |
|---|---|
| `model_usage`, `role_usage` | `init_sample_model_data()` (`run.py:1718`) takes seeds rather than clearing |
| `token_limit_usage` | `state._token_limit`'s accumulator |
| `cost` | `state._cost_limit`'s accumulator |
| `turns` | the `create_turn_limit()` node |
| `time` | the `create_time_limit()` node — `__enter__` derives `deadline = start + (limit − prior)` |
| `working_time` | the `create_working_limit()` node |
| `time`, `working_time` | also `SampleTiming`, via `init_sample_working_time()` — see §3 |

Seeding is done by a private helper in `_limit.py` operating on already-constructed nodes, invoked between construction and scope entry. It does **not** go on the factory signatures: `token_limit`, `cost_limit`, `turn_limit`, `time_limit`, and `working_limit` are all public API exported from `inspect_ai.util`, and `TaskState.__init__` is public too. This is framework-internal resume plumbing, not a user-facing capability.

`_TimeLimit` must be seeded before `__enter__`, since that is where the cancel-scope deadline is derived. The existing `_refresh_deadline()` (which re-derives the deadline when a live `ctl config` limit override changes) must account for the seed too, or an override would reset the sample to a full fresh budget.

Seeding `sample_model_usage` also restores `sample_total_tokens()`, which the `TokenInterval` trigger reads. That trigger measures tokens *since the last fire* against a reference captured on its first `tick()`, so it fires correctly either way — but its reported `sample_total_tokens` in `trigger_metadata` becomes honest again.

### 3. Log and reporting semantics

`SampleTiming` (`_util/working.py`) gains `prior_time` and `prior_working_time`, so that:

- `total_time = (now − start) + prior_time`
- `working_time = (now − start − waiting) + prior_working_time`

Note the second line is a change of derivation, not just an added term: `working_time` is currently computed as `total_time − sample_waiting_time()` (`run.py:2645`). With offsets in play it must be built from the un-offset elapsed, or `prior_time` is counted twice.

Two reporting surfaces then correct themselves, because both already read the seeded contextvar:

- `EvalSample.model_usage` (`run.py:2633`)
- the eval-level control-channel totals, via `_sample_usage()` (`run.py:1596`) → `record_sample_completed()`, which is documented as firing once per sample at its final outcome

Callers that use these values as *deltas* are unaffected by a constant offset — for example `Model._generate` brackets a call with `sample_working_time()` before and after (`_model.py:898`), and `_call_tools.py` does the same with `sample_waiting_time()`.

### 4. Edge cases

- **Prior usage already meets or exceeds the limit.** Only reachable when a limit is lowered between attempts (an `eval_retry` with different config, or a `ctl config` retune). For `_TimeLimit` this is sharp: a non-positive remaining budget makes `move_on_after()` fire immediately, cancelling the sample *inside `hydrate()`*, possibly mid-sandbox-restore. Seeds are therefore checked against the configured limits before the scopes are entered, raising a clean `LimitExceededError` up front rather than cancelling a partially restored sample.
- **`attempt == "resume_for_scoring"`.** Seed uniformly, but only §3 (the log totals) actually depends on it — there are no scoring-phase limits to seed. The sample limit scope opens at `run.py:1964` and has already exited by the time scoring runs, so scoring's *only* limit is the fresh `create_time_limit(time_limit / 2)` at `run.py:2219`. Token, cost, message, turn, and working limits are all out of scope, and each recorder no-ops on an empty tree (`record_model_usage`, `record_model_cost`, `record_waiting_time` all guard with `if node is None: return`). A model-graded scorer's tokens therefore still land in `sample_model_usage` — a plain contextvar, unaffected by the limit trees — and so reach the log and the eval totals, but are never charged against the token limit. Accounting and enforcement diverge here by design: scoring is deliberately not budgeted against the agent's allowance.
- **`scoring_time_limit` stays flat, at `time_limit / 2` of the *configured* limit.** Cumulative time accounting might suggest deriving it from what's left (`(time_limit − prior_elapsed) / 2`). Deliberately not done. The halving is not a budget slice; it is a bounded "give scoring a fair shot even though the clock ran out" allowance, and its whole purpose (`run.py:2199-2204`) is to survive the case where the agent has *already* fully exhausted its time on a hung container. Shrinking it by prior usage would give a twice-crashed sample a scoring window too short to reach that container — the exact failure the halving exists to bound. A resumed sample gets the same scoring allowance as a fresh one.
- **`_TimeLimit.__exit__`'s reported overage.** `value=self._end_time − self._start_time` must include the seed, or a `LimitExceededError` raised on a resumed sample understates by the prior elapsed time.
- **Requeue.** `_control/requeue.py` resumes through the same `_resume_if_checkpointed()` path, so it inherits this with no additional work.
- **In-process `retry_on_error` retries are deliberately unaffected.** That path recurses `task_run_sample` forwarding the *same* `resume_checkpoint` it was called with (`run.py:2456-2465`) — `None` for a sample on its first run. Such a retry therefore restores nothing today: not messages, not the sandbox, and (after this change) not usage either. Extending checkpoint restore to cover in-process retries is a separate question. The invariant this design holds to is narrower and easier to reason about: **a sample's usage is restored exactly when its state is** — both keyed on the same `resume_checkpoint`.

### 5. Consequences verified by prototype

The three items below were checked by patching `CheckpointUsage` into `schemas.py` on a scratch tree and running the real code; the patch was then reverted.

**No import cycle.** `_layout/schemas.py` is loaded early (`CheckpointEvent` imports `Checkpoint`, and the `Event` union imports `CheckpointEvent`), so a module-level `from inspect_ai.model._model_output import ModelUsage` there looked like a cycle risk — `_layout/__init__.py` documents exactly that hazard for `host_context`. It is not one: the import succeeds with `inspect_ai.event`, `inspect_ai.event._checkpoint`, `inspect_ai.util`, `inspect_ai.util._checkpoint`, `_layout.schemas`, `inspect_ai.model`, `inspect_ai.model._model_output`, `inspect_ai.log`, or `_eval.task.run` as the first module imported. `inspect_ai/__init__.py` fixes the order before anything reaches `schemas.py`. No lazy-module workaround is needed.

**This is a log-schema change and needs a coordinated ts-mono update.** `CheckpointEvent` *inherits* from `Checkpoint` (`event/_checkpoint.py:9`) and `from_details()` builds itself with `cls(**details.model_dump())`, so a `usage` field on `Checkpoint` appears on every `CheckpointEvent` in the `.eval` log. Regenerating confirms it: `python src/inspect_ai/_view/schema.py` produces a 55-line diff to `inspect-openapi.json`, adding a `CheckpointUsage` component and a `usage` property on `CheckpointEvent`. The `check-schema-and-types` CI job will fail until `generated.ts` is regenerated, so landing must follow [`.claude/skills/land-ts-mono/SKILL.md`](../.claude/skills/land-ts-mono/SKILL.md).

This is accepted rather than worked around. The alternatives are worse: a `Checkpoint` subclass used only for the file still leaks `usage` onto the event as an untyped extra (both models set `extra="allow"`), which puts data in logs that the schema doesn't describe; and a sidecar usage file breaks the invariant that the checkpoint file's existence *is* the commit point. Surfacing per-checkpoint usage in the log viewer is a genuine benefit besides.

One naming consequence: `CheckpointUsage`'s fields become public schema and TypeScript type names. `token_limit_usage` reads as internal — worth settling on a public-facing name during implementation, while keeping the distinction from `model_usage` explicit.

**Capture must tolerate having no limit trees.** `sample_limits()` raises `RuntimeError: No token limit node found. Is there a running sample?` when no sample scope is active. `tests/checkpoint/test_checkpointer.py` drives the *real* fire path with only `sample_state()` and `transcript()` patched (`_patch_sample_runtime`, line 118) — no limit trees. A bare `sample_limits()` call in `_fire_once()` would therefore break the existing unit tests. Capture goes through a helper that returns `None` when there is no live sample scope, and `usage` stays `CheckpointUsage | None`.

## Testing

- **`tests/checkpoint/test_schemas.py`** — round-trip `CheckpointUsage`; confirm a `ckpt-NNNNN.json` with no `usage` block parses with `usage is None`. (Both verified against the prototype.)
- **`tests/checkpoint/test_checkpointer.py`** — the existing fire-driving tests must keep passing with no limit trees in scope, and a fire with no live sample scope records `usage=None` rather than raising.
- **`tests/util/test_limit.py`** — a seeded node reports `prior + new`; a seeded `_TimeLimit` deadlines at `limit − prior` and reports the cumulative value in its `LimitExceededError`; a seed at or over the ceiling raises cleanly rather than cancelling; a `ctl config` override on a seeded time limit re-derives against the seed.
- **`tests/checkpoint/test_checkpoint_e2e.py`** — the existing `resume_kill_harness.py` already drives kill → resume → kill → resume with a real `SIGKILL` in a child process and a scripted mock model, so token counts are exact. Add: the final log's `model_usage`, `total_time`, and `working_time` span all attempts rather than only the last; and a token limit sized to span attempts trips *after* a resume, which today it never would.

This bug is only observable across a process death, so the e2e harness carries most of the weight.

## Documentation

- **`docs/checkpointing.qmd`** — the "Recovery" section lists what is restored (agent state, sandbox, events and store); add usage limits as a fourth item, stating that tokens, cost, turns, time, and working time continue from the checkpoint. The "Limitations" section gains the converse: work done between the last checkpoint and the crash is not counted, because it is rolled back — and restore time *is* charged to the time and working limits.
- **`CHANGELOG.md`** — one line under `## Unreleased`, outcome not mechanism.
- **Generated artifacts** — `python src/inspect_ai/_view/schema.py` regenerates `inspect-openapi.json`, and `generated.ts` must be regenerated in the ts-mono submodule to match. See §5; land via [`.claude/skills/land-ts-mono/SKILL.md`](../.claude/skills/land-ts-mono/SKILL.md).
