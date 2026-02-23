"""Example eval combining human_cli with tool_cli.

Gives the human user both `task` commands (submit, quit, score)
and `tools` commands (bash, etc.) inside the sandbox.
"""

from typing import Literal

import anyio

from inspect_ai import Task, task
from inspect_ai.agent import AgentState, agent, human_cli
from inspect_ai.agent._agent import Agent
from inspect_ai.tool import Tool, bash, run_tool_cli_service, text_editor
from inspect_ai.util import sandbox


@agent
def human_tools_agent(
    tools: list[Tool] | None = None,
    user: str | None = None,
    answer: bool | str = True,
    intermediate_scoring: bool = False,
) -> Agent:
    """Human CLI agent with tool access in the sandbox.

    Combines ``human_cli`` (task management commands) with ``tool_cli``
    (tool access commands) so the user gets both ``task`` and ``tools``
    commands in the sandbox shell.

    Args:
        tools: Tools to expose via the ``tools`` CLI command.
        user: User to login as.
        answer: Is an explicit answer required for this task?
        intermediate_scoring: Allow score checking while working.

    Returns:
        Agent that runs both services concurrently.
    """
    human = human_cli(
        answer=answer,
        intermediate_scoring=intermediate_scoring,
        user=user,
    )

    async def execute(state: AgentState) -> AgentState:
        done = False

        if tools:
            async with anyio.create_task_group() as tg:

                async def run_tool_service() -> None:
                    await run_tool_cli_service(
                        tools,
                        sandbox(),
                        until=lambda: done,
                        user=user,
                    )

                tg.start_soon(run_tool_service)

                try:
                    state = await human(state)
                finally:
                    done = True
        else:
            state = await human(state)

        return state

    return execute


@task
def human_with_tools(
    user: Literal["root", "nonroot"] | None = None,
) -> Task:
    """Task with human agent that has tool access in the sandbox."""
    return Task(
        solver=human_tools_agent(tools=[bash(), text_editor()], user=user),
        sandbox=("docker", "compose.yaml"),
    )
