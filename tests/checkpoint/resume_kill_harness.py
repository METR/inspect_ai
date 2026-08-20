"""Harness for the checkpoint resume-after-interrupt e2e test.

Lives alongside the e2e test in ``tests/checkpoint/`` and serves two roles:

1. **Importable** by the test — registering the ``@task`` / ``@modelapi`` so
   the final in-process resume works, and exposing shared constants.
2. **Runnable as a script** by the test for each interrupted attempt::

       python resume_kill_harness.py <log_dir> [<retry_from>]

Running the interrupted attempts in child processes is what lets the test
issue a *real* signal to an eval without taking down pytest. The ``crash``
tool ``os.kill``s its own (child) process with the signal named by
``SIGNAL_ENV``: ``SIGKILL`` for an unanticipated death (power loss / OOM /
preemption — no unwind, no log finalize), or ``SIGINT`` for what Ctrl-C
delivers.

Requires Docker (the sandbox backup path injects a Linux restic binary).
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

import anyio

from inspect_ai import Task, eval, eval_retry, task
from inspect_ai._util.working import report_sample_waiting_time
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.model._providers.mockllm import MockLLM
from inspect_ai.scorer import includes
from inspect_ai.tool import Tool, ToolChoice, ToolInfo, bash, tool
from inspect_ai.util import CheckpointConfig, TurnInterval, store

LAYER1_CONTENT = "plain1"
STORE_KEY = "answer"
SCRIPTED_MODEL = "scripteddecode/model"

# Write under $HOME (not /workspace) so the default-user home-dir auto-backup
# captures it — the task declares no `sandbox_paths`, exercising
# `resolve_sandbox_backup_paths` / `_resolve_home_and_cache`. Also drop a file
# under the XDG cache dir ($HOME/.cache) to prove auto-home mode excludes it.
WRITE_CMD = (
    'mkdir -p "$HOME/workspace/decoded" "$HOME/.cache" && '
    f"printf '{LAYER1_CONTENT}' > \"$HOME/workspace/decoded/layer1.txt\" && "
    'printf cache > "$HOME/.cache/junk.txt"'
)
# Written on each post-resume turn so the new snapshot has a non-empty diff vs
# its parent — used to assert file listing records the *changed* file.
RESUME_WRITE_CMD = 'printf resumed > "$HOME/workspace/resumed.txt"'

# The crash count + target live in a host file named by an env var, not module
# state: each killed attempt is a fresh process, and the count must survive
# both the kill and the next process's startup. The file is read by the
# scripted model (to decide when to crash vs work vs submit) and bumped by the
# crash tool just before it kills the process.
CANCEL_FILE_ENV = "INSPECT_TEST_RESUME_CANCEL_FILE"
TARGET_ENV = "INSPECT_TEST_RESUME_TARGET_CANCELS"

# Which signal the `crash` tool sends itself: SIGKILL (unanticipated death, no
# unwind) or SIGINT (what Ctrl-C delivers — graceful cancel, log finalized,
# sandboxes torn down). Resume must work from either.
SIGNAL_ENV = "INSPECT_TEST_RESUME_SIGNAL"

# Knobs the usage-restoration tests vary per attempt (each attempt is a fresh
# process, so they travel as environment rather than arguments):
#
# - whether the task opts in to continuing its usage counters on resume:
#   unset omits `restore_usage` from `CheckpointConfig` entirely (the literal
#   default), "0"/"1" pass it explicitly as False/True,
# - a sample token limit (unset = no limit), and
# - seconds of provider waiting time the scripted model reports on the first
#   attempt (see `_report_model_waiting`).
RESTORE_USAGE_ENV = "INSPECT_TEST_RESTORE_USAGE"
TOKEN_LIMIT_ENV = "INSPECT_TEST_TOKEN_LIMIT"
MODEL_WAITING_ENV = "INSPECT_TEST_MODEL_WAITING_SECONDS"


def crash_signal() -> signal.Signals:
    return signal.Signals[os.environ.get(SIGNAL_ENV, "SIGKILL")]


def cancels_done() -> int:
    f = os.environ.get(CANCEL_FILE_ENV)
    return int(Path(f).read_text() or "0") if f and Path(f).exists() else 0


def bump_cancels() -> int:
    n = cancels_done() + 1
    f = os.environ.get(CANCEL_FILE_ENV)
    if f:
        Path(f).write_text(str(n))
    return n


def target_cancels() -> int:
    return int(os.environ.get(TARGET_ENV, "1"))


class _ResumeState:
    """In-process model-call counter.

    Meaningful only for the resume that runs in the test process; killed
    attempts run in their own process.
    """

    generates: int = 0


_resume_state = _ResumeState()


def generates() -> int:
    return _resume_state.generates


def reset_generates() -> None:
    _resume_state.generates = 0


@tool
def remember() -> Tool:
    async def execute(key: str, value: str) -> str:
        """Record a key/value note in the sample store.

        Args:
            key: short label for the note.
            value: the value to remember.

        Returns:
            Confirmation string.
        """
        store().set(key, value)
        return f"remembered: {key}"

    return execute


@tool
def crash() -> Tool:
    async def execute() -> str:
        """Signal the eval process to die (SIGKILL) or cancel (SIGINT)."""
        # Record the crash before signalling (flushed to disk), then signal our
        # own process. Running inside the child, this is the child's PID.
        bump_cancels()
        os.kill(os.getpid(), crash_signal())
        # SIGKILL never gets here; under SIGINT this is where the eval's
        # cancellation lands.
        await anyio.sleep_forever()
        return "crashed"

    return execute


# eval_retry reconstructs the task by registry name and rebuilds the model by
# name from the log — so the task must be a registered @task and the scripted
# behavior must live in a registered model provider. The provider drives a
# linear script keyed off the number of completed tool turns in the restored
# conversation, plus the host-file crash count:
#
#   turn 0: bash (write a sandbox file)        -> ckpt-1 fires next turn
#   turn 1: remember (write the store)         -> ckpt-2 fires next turn
#   turn 2: crash (1st kill) ..................... SIGKILL, then resume
#           bash (write a new sandbox file)    -> ckpt-3 fires next turn
#   turn 3: crash (2nd kill) ..................... SIGKILL, then resume
#           bash (write a new sandbox file)    -> ckpt-4 fires next turn
#   turn 4: submit
#
# Each resume cycle does one work turn (so a fresh checkpoint commits) before
# crashing, until `target` crashes are reached; the final resume submits.


def _report_model_waiting() -> None:
    """Report `MODEL_WAITING_ENV` seconds of provider waiting time.

    Real providers report rate-limit / dispatch waits through this same
    call; reporting it here (without actually sleeping) is what makes a
    sample's total time and its working time differ by a known amount, so
    a resume that continues both can be checked for confusing one with the
    other. No-op unless the env var is set.
    """
    seconds = float(os.environ.get(MODEL_WAITING_ENV) or 0.0)
    if seconds:
        report_sample_waiting_time(seconds)


def _scripted_outputs(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: GenerateConfig,
) -> ModelOutput:
    _resume_state.generates += 1
    n = sum(1 for m in input if isinstance(m, ChatMessageTool))
    if n == 1:
        # turn 1 only ever runs on the first attempt (later attempts restore
        # the conversation), and the checkpoint that closes it captures the
        # wait — so it lands in prior usage, never in the resume's own.
        _report_model_waiting()
    output = _scripted_call(n)
    # MockLLM only fills usage in for its *iterator* form — a callable
    # `custom_outputs` is returned straight to the caller — so report it here
    # or the sample records no token usage at all. One token per message in
    # the (deterministic) conversation, plus one for the reply.
    output.usage = ModelUsage(
        input_tokens=len(input), output_tokens=1, total_tokens=len(input) + 1
    )
    return output


def _scripted_call(n: int) -> ModelOutput:
    """The tool call scripted for a conversation with `n` completed tool turns."""
    done = cancels_done()
    target = target_cancels()
    if n == 0:
        return ModelOutput.for_tool_call(SCRIPTED_MODEL, "bash", {"command": WRITE_CMD})
    if n == 1:
        return ModelOutput.for_tool_call(
            SCRIPTED_MODEL, "remember", {"key": STORE_KEY, "value": LAYER1_CONTENT}
        )
    # The k-th crash (k = done) lands at turn `2 + done`; cycles that don't
    # crash do a work turn that writes a new sandbox file (non-empty diff vs
    # parent), then the final resume submits.
    if done < target and n == 2 + done:
        return ModelOutput.for_tool_call(SCRIPTED_MODEL, "crash", {})
    if n < 2 + target:
        return ModelOutput.for_tool_call(
            SCRIPTED_MODEL, "bash", {"command": RESUME_WRITE_CMD}
        )
    return ModelOutput.for_tool_call(
        SCRIPTED_MODEL, "submit", {"answer": LAYER1_CONTENT}
    )


@modelapi(name="scripteddecode")
def _scripteddecode_provider() -> type[MockLLM]:
    class ScriptedDecode(MockLLM):
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            # ignore any persisted custom_outputs; drive from _scripted_outputs
            kwargs.pop("custom_outputs", None)
            super().__init__(model_name, custom_outputs=_scripted_outputs, **kwargs)

    return ScriptedDecode


@task
def resume_decode_task() -> Task:
    restore_usage_env = os.environ.get(RESTORE_USAGE_ENV)
    checkpoint = (
        # Unset: never pass `restore_usage`, so the config layer sees the
        # literal default — the path a caller who never heard of this
        # option takes.
        CheckpointConfig(
            trigger=TurnInterval(every=1),
            # No sandbox_paths: the default sandbox's $HOME is auto-captured.
            retention="retain",
        )
        if restore_usage_env is None
        else CheckpointConfig(
            trigger=TurnInterval(every=1),
            retention="retain",
            restore_usage=restore_usage_env == "1",
        )
    )
    return Task(
        dataset=[Sample(id="resume", input="decode the layers", target=LAYER1_CONTENT)],
        solver=react(tools=[bash(timeout=60), remember(), crash()]),
        scorer=includes(),
        # Default sandbox image: its ~955 MB /root is mostly /root/.cache,
        # which auto-home mode excludes — so the egress stays small without a
        # custom small-home image, and this exercises that exclude for real.
        sandbox="docker",
        checkpoint=checkpoint,
        token_limit=int(limit) if (limit := os.environ.get(TOKEN_LIMIT_ENV)) else None,
    )


def run_eval(log_dir: str, retry_from: str | None = None) -> None:
    """Run a fresh eval, or resume one from a prior log.

    Never returns when the scripted run is due to crash — the ``crash`` tool
    ``SIGKILL``s the process.
    """
    if retry_from is None:
        eval(resume_decode_task(), model=SCRIPTED_MODEL, log_dir=log_dir)
    else:
        eval_retry(read_eval_log(retry_from), log_dir=log_dir)


def main() -> None:
    log_dir = sys.argv[1]
    retry_from = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    run_eval(log_dir, retry_from)


if __name__ == "__main__":
    main()
