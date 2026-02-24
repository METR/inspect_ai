from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import Setting, TaskState, Workspace, setting
from inspect_ai.solver._task_state import set_sample_state
from inspect_ai.tool import tool


@tool
def addition():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x (int): First number to add.
            y (int): Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y

    return execute


def test_setting_defaults():
    s = Setting()
    assert s.workspaces == ()
    assert s.tools == ()
    assert s.on_turn is None


def test_workspace_defaults():
    ws = Workspace()
    assert ws.sandbox == "default"
    assert ws.description == ""
    assert ws.user is None


def test_workspace_with_all_fields():
    ws = Workspace(sandbox="main", description="Primary workspace", user="hacker")
    assert ws.sandbox == "main"
    assert ws.description == "Primary workspace"
    assert ws.user == "hacker"


def test_setting_with_workspaces():
    s = Setting(
        workspaces=(
            Workspace(sandbox="default", description="Workspace", user="user"),
            Workspace(sandbox="db", description="Database", user="postgres"),
        ),
    )
    assert len(s.workspaces) == 2
    assert s.workspaces[0].sandbox == "default"
    assert s.workspaces[1].sandbox == "db"


def test_setting_accessor_returns_none_when_no_state():
    assert setting() is None


def test_task_state_setting_property():
    s = Setting(workspaces=(Workspace(description="test"),))
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id=0,
        epoch=1,
        input="test",
        messages=[ChatMessageUser(content="test")],
        setting=s,
    )
    assert state.setting is s

    # setter works
    s2 = Setting()
    state.setting = s2
    assert state.setting is s2


def test_setting_accessor_returns_setting_from_state():
    s = Setting(tools=(addition(),))
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id=0,
        epoch=1,
        input="test",
        messages=[ChatMessageUser(content="test")],
        setting=s,
    )
    set_sample_state(state)
    result = setting()
    assert result is s
    assert len(result.tools) == 1
