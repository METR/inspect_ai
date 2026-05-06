"""Tests that scoring errors after operator interrupts preserve interrupt context.

When a sample is interrupted by an operator and scoring then fails (e.g.
because setup didn't complete), the error message should indicate that
the scoring failure happened after an interrupt — not just show the scorer's
TypeError with no context about why setup state was missing.
"""

import anyio

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log._samples import sample_active
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, TaskState, solver


@solver
def interrupt_then_sleep_solver():
    """Trigger an interrupt and yield so the CancelledError is delivered."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        active = sample_active()
        if active is not None:
            active.interrupt("score")
        # Yield to allow the cancellation to be delivered
        await anyio.sleep(1)
        return state

    return solve


@scorer(metrics=[])
def failing_scorer():
    """Scorer that crashes on None store values, simulating incomplete setup."""

    async def score(state: TaskState, target: Target) -> Score:
        value = None
        float(value)  # type: ignore
        return Score(value=0)  # unreachable

    return score


def test_scoring_error_after_interrupt_includes_context():
    """Scoring error after operator interrupt should mention the interrupt."""
    task = Task(
        dataset=[Sample(input="test", target="target")],
        solver=[interrupt_then_sleep_solver()],
        scorer=failing_scorer(),
    )

    log = eval(task, model="mockllm/model")[0]

    assert log.samples is not None
    assert len(log.samples) == 1

    sample = log.samples[0]
    assert sample.error is not None
    assert "operator interrupt" in sample.error.message.lower(), (
        f"Expected error to mention operator interrupt, got: {sample.error.message}"
    )
