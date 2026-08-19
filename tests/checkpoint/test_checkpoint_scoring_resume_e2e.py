"""End-to-end scoring-phase checkpoint-resume test.

The agent finishes cleanly, the scorer hard-kills the process, then retry
skips the agent and re-runs scoring to success.

This is the scenario the scoring-phase resume machinery exists for. The agent
loop runs to a clean ``submit`` — so the checkpointer fires a final
``agent_complete`` checkpoint — and then the scorer ``SIGKILL``s its own
process *before* scoring commits. On retry, inspect reads the latest
``agent_complete`` checkpoint, tags the sample
``"resume_for_scoring"``, and the ``react`` agent fast-path-returns its
restored state with **zero** model calls; the scorer then re-runs to success.

Like the mid-agent sibling (``test_checkpoint_e2e.py``), a real ``SIGKILL``
can't kill the pytest process and let it continue, so the killed attempt runs
in a **child process** (the harness in ``resume_scoring_kill_harness.py``, run
as a script); the scorer kills that child. The final resume runs in-process.

Requires Docker: the sandbox backup path injects/execs a Linux restic binary
inside the sandbox, which only works with a Linux container.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from test_helpers.utils import flaky_retry, skip_if_no_docker

from checkpoint.resume_scoring_kill_harness import (
    AGENT_SLEEP_ENV,
    ANSWER,
    CANCEL_FILE_ENV,
    RESTORE_USAGE_ENV,
    TARGET_ENV,
    TIME_LIMIT_ENV,
    WORKING_LIMIT_ENV,
    generates,
    reset_generates,
)
from inspect_ai import eval_retry
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.scorer import CORRECT


def _latest_log(log_dir: str) -> str:
    """Location of the most recently written eval log (timestamp-prefixed)."""
    logs = list_eval_logs(log_dir)
    assert logs, f"no eval logs under {log_dir}"
    return max(logs, key=lambda info: info.name).name


def _run_killed_attempt(log_dir: str, retry_from: str | None, tests_dir: Path) -> None:
    """Run an eval in a child process whose scorer ``SIGKILL``s itself.

    Asserts the child died by signal rather than exiting normally.
    """
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(tests_dir), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    harness = str(tests_dir / "checkpoint" / "resume_scoring_kill_harness.py")
    proc = subprocess.run(
        [sys.executable, harness, log_dir, retry_from or ""],
        env=env,
        timeout=600,
    )
    assert proc.returncode == -signal.SIGKILL, (
        f"expected the child to die by SIGKILL (-{signal.SIGKILL}); "
        f"got returncode {proc.returncode}"
    )


def _inspect_projects() -> set[str]:
    """Names of inspect docker compose projects currently known to docker."""
    result = subprocess.run(
        ["docker", "compose", "ls", "--all", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    try:
        projects = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return set()
    return {
        p.get("Name", "") for p in projects if p.get("Name", "").startswith("inspect-")
    }


def _force_remove_project(name: str) -> None:
    """Best-effort force-remove the containers of a leaked compose project."""
    ids = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={name}"],
        capture_output=True,
        text=True,
    ).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)


@skip_if_no_docker
@pytest.mark.slow
def test_checkpoint_scoring_phase_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scoring-crash count (host file) + target are inherited by the child.
    cancel_file = tmp_path / "cancels.txt"
    monkeypatch.setenv(CANCEL_FILE_ENV, str(cancel_file))
    monkeypatch.setenv(TARGET_ENV, "1")
    # The crash count is stateful on disk. Under flaky-retry (this test is
    # `_needs_flaky_retry` via `skip_if_no_docker`) the body re-runs with the
    # same `tmp_path`, so reset it — otherwise a retry would inherit a
    # count >= target, the scorer would never crash, and the retry would
    # spuriously fail.
    cancel_file.unlink(missing_ok=True)

    log_dir = str(tmp_path / "logs")
    tests_dir = Path(__file__).parent.parent

    # A hard kill skips sandbox teardown, so the killed attempt leaks its
    # sandbox container. Track inspect projects before/after and force-remove
    # the ones this test leaks (the final resume cleans up its own).
    projects_before = _inspect_projects()
    try:
        # --- attempt #0: fresh eval; agent completes, scorer hard-kills ------
        _run_killed_attempt(log_dir, None, tests_dir)

        # --- final resume: runs in this process, scoring-phase only ----------
        reset_generates()
        resume = eval_retry(read_eval_log(_latest_log(log_dir)), log_dir=log_dir)[0]
    finally:
        for name in _inspect_projects() - projects_before:
            _force_remove_project(name)

    assert resume.status == "success"
    assert resume.samples is not None and len(resume.samples) == 1
    sample = resume.samples[0]
    assert sample.error is None

    # Headline: the agent loop was skipped entirely on the scoring-phase
    # resume — zero model calls. A plain RETRY (or a from-scratch rerun) would
    # have driven the scripted model again.
    assert generates() == 0

    # The restored agent output was re-scored to success.
    assert sample.scores is not None
    assert sample.scores["crashing_includes"].value == CORRECT
    assert ANSWER in sample.output.completion


# The killed attempt burns this much sample time before its agent finishes;
# the resume then runs under a *lower* limit, so a resume that continued the
# prior attempt's clock would open its time scope already expired.
_AGENT_SLEEP_SECONDS = 30
_RESUME_TIME_LIMIT_SECONDS = 25


@skip_if_no_docker
@pytest.mark.slow
@flaky_retry(max_retries=1)
def test_scoring_resume_keeps_its_full_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scoring-only resume keeps its whole time budget, even with usage restored.

    Tokens, cost and turns are checked cooperatively when they are recorded,
    but time and working time enforce through a cancel-scope deadline and a
    background poller that fire whether or not the agent is running. A
    scoring-phase resume that continued the prior attempt's clock would
    therefore cancel the checkpoint restore running inside its own solver
    task — and the agent it would be protecting already finished.

    The time limit is lowered between attempts (a retry can be configured
    differently) so the prior attempt's clock exceeds it by construction.
    """
    cancel_file = tmp_path / "cancels.txt"
    monkeypatch.setenv(CANCEL_FILE_ENV, str(cancel_file))
    monkeypatch.setenv(TARGET_ENV, "1")
    monkeypatch.setenv(RESTORE_USAGE_ENV, "1")
    monkeypatch.setenv(AGENT_SLEEP_ENV, str(_AGENT_SLEEP_SECONDS))
    monkeypatch.delenv(TIME_LIMIT_ENV, raising=False)
    monkeypatch.delenv(WORKING_LIMIT_ENV, raising=False)

    log_dir = str(tmp_path / "logs")
    # Both are stateful on disk and flaky-retry re-runs this body with the
    # same `tmp_path`: a stale crash count would skip the kill, and stale
    # checkpoints would be resumed instead of this run's own.
    cancel_file.unlink(missing_ok=True)
    shutil.rmtree(log_dir, ignore_errors=True)

    tests_dir = Path(__file__).parent.parent

    projects_before = _inspect_projects()
    try:
        _run_killed_attempt(log_dir, None, tests_dir)
        monkeypatch.setenv(TIME_LIMIT_ENV, str(_RESUME_TIME_LIMIT_SECONDS))
        # The working-time limit enforces via the same kind of background
        # poller as the time limit, independent of agent activity, but is
        # inert here unless the killed attempt's working time is also seeded
        # on resume — pins the other half of `_clock_seed_usage`.
        monkeypatch.setenv(WORKING_LIMIT_ENV, str(_RESUME_TIME_LIMIT_SECONDS))
        reset_generates()
        resume = eval_retry(read_eval_log(_latest_log(log_dir)), log_dir=log_dir)[0]
    finally:
        for name in _inspect_projects() - projects_before:
            _force_remove_project(name)

    assert resume.status == "success"
    assert resume.samples is not None and len(resume.samples) == 1
    sample = resume.samples[0]
    assert sample.error is None

    # Headline: no limit was stamped. Seeding the prior clock here would have
    # cancelled the restore and logged a time limit against a sample whose
    # agent completed cleanly.
    assert sample.limit is None, f"scoring resume was stopped by {sample.limit}"
    assert generates() == 0
    assert sample.scores is not None
    assert sample.scores["crashing_includes"].value == CORRECT
    assert ANSWER in sample.output.completion

    # ...and the premise held: the clock a seeding resume would have started
    # from — this sample's cumulative time less the resume's own elapsed time,
    # `started_at` being per-attempt — really did exceed the resume's budget.
    assert sample.total_time is not None
    assert sample.started_at and sample.completed_at
    resume_elapsed = (
        datetime.fromisoformat(sample.completed_at)
        - datetime.fromisoformat(sample.started_at)
    ).total_seconds()
    prior_time = sample.total_time - resume_elapsed
    assert prior_time > _RESUME_TIME_LIMIT_SECONDS, (
        f"the killed attempt only reached {prior_time}s, so this no longer "
        f"exercises a resume whose inherited clock would exceed its limit"
    )
    # ...and the lowered limit actually reached the resume. Without this, a
    # resume that silently dropped the limit would also pass with
    # `sample.limit is None` above, for the wrong reason.
    assert resume.eval.config.time_limit == _RESUME_TIME_LIMIT_SECONDS
