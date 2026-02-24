from __future__ import annotations

from typing import Awaitable, Callable, NamedTuple

from inspect_ai.tool._tool import Tool
from inspect_ai.tool._tool_def import ToolDef

OnTurn = Callable[[], Awaitable[bool | str | None]]
"""Callback fired each iteration of the agent loop.

Returns:
    `False` to stop the agent (sets `completed = True`).
    `str` to inject a user message and continue (skips on_continue).
    `None` or `True` to continue normally.
"""


class Workspace(NamedTuple):
    """A sandbox environment the agent should work in."""

    sandbox: str = "default"
    """Sandbox environment name (matches docker-compose service name)."""

    description: str = ""
    """Human-readable description of this workspace for the agent."""

    user: str | None = None
    """User to run commands as in this sandbox."""


class Setting(NamedTuple):
    """Execution setting provided by the problem definition.

    Lets the task communicate workspaces, tools, and per-turn callbacks
    to agent scaffolding.
    """

    workspaces: tuple[Workspace, ...] = ()
    """Sandboxes the agent should work in. First is primary.
    Non-workspace sandboxes (targets, resources) are omitted."""

    tools: tuple[Tool | ToolDef, ...] = ()
    """Custom tools the agent needs (CLIs, binaries, etc.).
    Standard tools (bash, editor) are created by scaffolding from workspace metadata."""

    on_turn: OnTurn | None = None
    """Callback fired each iteration of the agent loop."""


def setting() -> Setting | None:
    """Get the Setting for the current sample, if any."""
    from ._task_state import sample_state

    state = sample_state()
    if state is None:
        return None
    return state._setting
