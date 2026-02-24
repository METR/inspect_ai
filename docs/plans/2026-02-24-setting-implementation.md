# Setting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `SampleContext`/`SandboxRole` with `Setting`/`Workspace` and add workspace-based tool creation in scaffolding.

**Architecture:** This is a rename+reshape of the existing SampleContext implementation. `SampleContext` becomes `Setting`, `SandboxRole` becomes `Workspace` (now carries the sandbox name, drops the role enum), and scaffolding gains workspace-based bash tool creation. See `docs/plans/2026-02-24-setting-design.md` for the full design.

**Tech Stack:** Python, NamedTuple, pytest, mockllm

---

### Task 1: Create `_setting.py` with Workspace and Setting types

**Files:**
- Create: `src/inspect_ai/solver/_setting.py`
- Test: `tests/solver/test_setting.py`

**Step 1: Write the failing tests for Workspace and Setting**

Create `tests/solver/test_setting.py`:

```python
from inspect_ai.solver import Setting, Workspace, setting


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/solver/test_setting.py -v`
Expected: FAIL — `ImportError: cannot import name 'Setting'`

**Step 3: Write `_setting.py`**

Create `src/inspect_ai/solver/_setting.py`:

```python
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
```

**Step 4: Update `src/inspect_ai/solver/__init__.py` exports**

Replace the `_sample_context` import line with:

```python
from ._setting import OnTurn, Setting, Workspace, setting
```

And in `__all__`, replace `"OnTurn", "SampleContext", "SandboxRole", "sample_context"` with:

```python
"OnTurn",
"Setting",
"Workspace",
"setting",
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/solver/test_setting.py -v`
Expected: PASS (all 5 tests)

**Step 6: Commit**

```bash
git add src/inspect_ai/solver/_setting.py src/inspect_ai/solver/__init__.py tests/solver/test_setting.py
git commit -m "feat: add Setting and Workspace types (replaces SampleContext/SandboxRole)"
```

---

### Task 2: Update TaskState to use Setting

**Files:**
- Modify: `src/inspect_ai/solver/_task_state.py`
- Test: `tests/solver/test_setting.py`

**Step 1: Write the failing test**

Add to `tests/solver/test_setting.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/solver/test_setting.py::test_task_state_setting_property -v`
Expected: FAIL — `TypeError: TaskState.__init__() got an unexpected keyword argument 'setting'`

**Step 3: Update `_task_state.py`**

In the `TYPE_CHECKING` import block, change:

```python
if TYPE_CHECKING:
    from ._sample_context import SampleContext
```

to:

```python
if TYPE_CHECKING:
    from ._setting import Setting
```

In `__init__` signature, change `sample_context: "SampleContext | None" = None` to:

```python
setting: "Setting | None" = None,
```

In `__init__` body, change `self._sample_context: SampleContext | None = sample_context` to:

```python
self._setting: Setting | None = setting
```

Replace the `sample_context` property and setter with:

```python
@property
def setting(self) -> "Setting | None":
    """Execution setting provided by the problem definition."""
    return self._setting

@setting.setter
def setting(self, val: "Setting | None") -> None:
    self._setting = val
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/solver/test_setting.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/inspect_ai/solver/_task_state.py tests/solver/test_setting.py
git commit -m "feat: update TaskState to use Setting instead of SampleContext"
```

---

### Task 3: Update Task and eval loop

**Files:**
- Modify: `src/inspect_ai/_eval/task/task.py`
- Modify: `src/inspect_ai/_eval/task/run.py`
- Test: `tests/solver/test_setting.py`

**Step 1: Write the failing tests**

Add to `tests/solver/test_setting.py`:

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes


def test_task_with_static_setting():
    s = Setting(workspaces=(Workspace(description="test"),))
    task = Task(
        dataset=[Sample(input="test", target="test")],
        setting=s,
        scorer=includes(),
    )
    assert task.setting is s


def test_task_with_factory_setting():
    def make_setting(sample: Sample) -> Setting:
        return Setting(tools=(addition(),))

    task = Task(
        dataset=[Sample(input="test", target="test")],
        setting=make_setting,
        scorer=includes(),
    )
    assert callable(task.setting)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/solver/test_setting.py::test_task_with_static_setting -v`
Expected: FAIL — `TypeError: Task.__init__() got an unexpected keyword argument 'setting'`

**Step 3: Update `task.py`**

Change the import from:

```python
from inspect_ai.solver._sample_context import SampleContext
```

to:

```python
from inspect_ai.solver._setting import Setting
```

In `Task.__init__` signature, change `sample_context: SampleContext | Callable[[Sample], SampleContext] | None = None` to:

```python
setting: Setting | Callable[[Sample], Setting] | None = None,
```

In `Task.__init__` body, change `self.sample_context = sample_context` to:

```python
self.setting = setting
```

In the docstring, change the `sample_context` entry to:

```
setting: Execution setting for samples. Provides workspaces, custom
    tools, and per-turn callbacks to agent scaffolding. Pass a `Setting`
    for all samples or a callable that takes a `Sample` and returns a
    `Setting` for per-sample configuration.
```

**Step 4: Update `run.py`**

In the `TYPE_CHECKING` import block, change:

```python
if TYPE_CHECKING:
    from inspect_ai.solver._sample_context import SampleContext
```

to:

```python
if TYPE_CHECKING:
    from inspect_ai.solver._setting import Setting
```

In `resolve_dataset` signature, change `task_sample_context` to:

```python
task_setting: "Setting | Callable[[Sample], Setting] | None" = None,
```

In `resolve_dataset` body, change the sample_context application block to:

```python
# apply setting after deepcopy (avoids deepcopy of callables)
if task_setting is not None:
    for state, sample in zip(states, samples):
        if callable(task_setting):
            state.setting = task_setting(sample)
        else:
            state.setting = task_setting
```

In `task_run`, change `task_sample_context=task.sample_context` to:

```python
task_setting=task.setting,
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/solver/test_setting.py -v`
Expected: PASS (all 9 tests)

**Step 6: Commit**

```bash
git add src/inspect_ai/_eval/task/task.py src/inspect_ai/_eval/task/run.py tests/solver/test_setting.py
git commit -m "feat: update Task and eval loop to use Setting"
```

---

### Task 4: Update react() to use Setting + workspace tools

**Files:**
- Modify: `src/inspect_ai/agent/_react.py`
- Test: `tests/solver/test_setting.py`

**Step 1: Write the failing integration test**

Add to `tests/solver/test_setting.py`:

```python
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import basic_agent, system_message


def test_setting_workspace_creates_bash_in_basic_agent():
    """Test that workspaces cause scaffolding to create bash tools."""
    s = Setting(
        workspaces=(Workspace(sandbox="default", description="Workspace", user="testuser"),),
    )
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target=["2", "2.0"])],
        setting=s,
        solver=basic_agent(
            tools=[],
            message_limit=5,
        ),
        scorer=includes(),
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "2"},
            )
        ],
    )

    log = eval(task, model=model)[0]
    assert log.status == "success"

    model_event = next(
        event for event in log.samples[0].transcript.events if event.event == "model"
    )
    tool_names = {t.name for t in model_event.tools}
    assert "bash" in tool_names
    assert "submit" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/solver/test_setting.py::test_setting_workspace_creates_bash_in_basic_agent -v`
Expected: FAIL — `bash` tool not present (only `submit`)

**Step 3: Update `_react.py`**

Change the import:

```python
from inspect_ai.solver._sample_context import sample_context as get_sample_context
```

to:

```python
from inspect_ai.solver._setting import setting as get_setting
```

Add import for bash at the top (alongside other tool imports):

```python
from inspect_ai.tool._tools._execute import bash
```

Update `_merge_sample_tools`:

```python
def _merge_sample_tools(
    tools: list[Tool | ToolDef | ToolSource],
) -> list[Tool | ToolDef | ToolSource]:
    """Prepend setting tools (and workspace bash tools) before solver tools, deduplicating by name."""
    s = get_setting()
    if s is None:
        return tools

    # collect setting tools + workspace bash tools
    setting_tools: list[Tool | ToolDef] = list(s.tools)
    for ws in s.workspaces:
        setting_tools.append(bash(sandbox=ws.sandbox, user=ws.user))

    if not setting_tools:
        return tools

    # build a set of setting tool names
    setting_tool_names: set[str] = set()
    for st in setting_tools:
        setting_tool_names.add(
            ToolDef(st).name if not isinstance(st, ToolDef) else st.name
        )

    # filter out solver tools that conflict with setting tools
    filtered: list[Tool | ToolDef | ToolSource] = []
    for solver_tool in tools:
        if isinstance(solver_tool, ToolSource):
            filtered.append(solver_tool)
        else:
            name = (
                ToolDef(solver_tool).name
                if not isinstance(solver_tool, ToolDef)
                else solver_tool.name
            )
            if name not in setting_tool_names:
                filtered.append(solver_tool)

    return list(setting_tools) + filtered
```

Update `_call_on_turn_callback`:

```python
async def _call_on_turn_callback() -> bool | str | None:
    """Call the setting on_turn callback if present."""
    s = get_setting()
    if s is None or s.on_turn is None:
        return None
    return await s.on_turn()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/solver/test_setting.py -v`
Expected: PASS (all 10 tests)

**Step 5: Run ruff**

Run: `uv run ruff check --fix src/inspect_ai/agent/_react.py && uv run ruff format src/inspect_ai/agent/_react.py`

**Step 6: Commit**

```bash
git add src/inspect_ai/agent/_react.py tests/solver/test_setting.py
git commit -m "feat: update react() to use Setting + workspace bash tools"
```

---

### Task 5: Update basic_agent() to use Setting + workspace tools

**Files:**
- Modify: `src/inspect_ai/solver/_basic_agent.py`

**Step 1: Verify existing integration test passes with react, fails with basic_agent**

The test `test_setting_workspace_creates_bash_in_basic_agent` from Task 4 already uses `basic_agent`. Run it:

Run: `uv run pytest tests/solver/test_setting.py::test_setting_workspace_creates_bash_in_basic_agent -v`
Expected: Should already PASS if basic_agent uses the same merge function. If not, continue.

**Step 2: Update `_basic_agent.py`**

Change the import:

```python
from ._sample_context import sample_context as get_sample_context
```

to:

```python
from ._setting import setting as get_setting
```

Add import for bash:

```python
from inspect_ai.tool._tools._execute import bash
```

In `basic_agent_loop`, replace the tool merging block:

```python
# merge setting tools into state.tools
s = get_setting()
if s is not None:
    from inspect_ai.tool._tool_def import ToolDef

    # collect setting tools + workspace bash tools
    setting_tools: list[Tool | ToolDef] = list(s.tools)
    for ws in s.workspaces:
        setting_tools.append(bash(sandbox=ws.sandbox, user=ws.user))

    if setting_tools:
        setting_tool_names: set[str] = set()
        for st in setting_tools:
            setting_tool_names.add(
                ToolDef(st).name if not isinstance(st, ToolDef) else st.name
            )
        filtered = [
            t
            for t in state.tools
            if ToolDef(t).name not in setting_tool_names
        ]
        merged: list[Tool] = [
            t if isinstance(t, Tool) else t.as_tool() for t in setting_tools
        ]
        merged.extend(filtered)
        state.tools = merged
```

Replace the on_turn block at the bottom of the loop:

```python
# fire setting on_turn callback
on_turn_s = get_setting()
if on_turn_s is not None and on_turn_s.on_turn is not None:
    on_turn_result = await on_turn_s.on_turn()
    if on_turn_result is False:
        state.completed = True
        break
    elif isinstance(on_turn_result, str):
        state.messages.append(
            ChatMessageUser(content=on_turn_result)
        )
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/solver/test_setting.py tests/solver/test_basic_agent.py -v`
Expected: PASS (all tests in both files)

**Step 4: Run ruff**

Run: `uv run ruff check --fix src/inspect_ai/solver/_basic_agent.py && uv run ruff format src/inspect_ai/solver/_basic_agent.py`

**Step 5: Commit**

```bash
git add src/inspect_ai/solver/_basic_agent.py
git commit -m "feat: update basic_agent() to use Setting + workspace bash tools"
```

---

### Task 6: Update top-level exports and add remaining tests

**Files:**
- Modify: `src/inspect_ai/__init__.py`
- Test: `tests/solver/test_setting.py`

**Step 1: Update `src/inspect_ai/__init__.py`**

Change:

```python
from inspect_ai.solver._sample_context import SampleContext, SandboxRole
```

to:

```python
from inspect_ai.solver._setting import Setting, Workspace
```

In `__all__`, change `"SampleContext", "SandboxRole"` to:

```python
"Setting",
"Workspace",
```

**Step 2: Add remaining integration tests**

Add to `tests/solver/test_setting.py`:

```python
def test_setting_tools_merged_into_basic_agent():
    """Test that Setting.tools are merged into basic_agent."""
    s = Setting(tools=(addition(),))
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target=["2", "2.0"])],
        setting=s,
        solver=basic_agent(
            init=system_message(
                "You are a helpful assistant. Call submit() when done."
            ),
            tools=[],
            message_limit=5,
        ),
        scorer=includes(),
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "2"},
            )
        ],
    )

    log = eval(task, model=model)[0]
    assert log.status == "success"

    model_event = next(
        event for event in log.samples[0].transcript.events if event.event == "model"
    )
    tool_names = {t.name for t in model_event.tools}
    assert "addition" in tool_names
    assert "submit" in tool_names


def test_setting_tool_dedup():
    """Test that setting tools override solver tools with the same name."""

    @tool(name="addition")
    def custom_addition():
        async def execute(x: int, y: int):
            """Add numbers but returns wrong answer.

            Args:
                x (int): First number.
                y (int): Second number.
            """
            return x + y + 100

        return execute

    s = Setting(tools=(addition(),))
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target=["2", "2.0"])],
        setting=s,
        solver=basic_agent(
            tools=[custom_addition()],
            message_limit=5,
        ),
        scorer=includes(),
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "2"},
            )
        ],
    )

    log = eval(task, model=model)[0]
    assert log.status == "success"


def test_setting_on_turn_stops():
    """Test that on_turn returning False stops the agent."""
    call_count = 0

    async def stop_after_two() -> bool | str | None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            return False
        return None

    s = Setting(on_turn=stop_after_two)
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target="2")],
        setting=s,
        solver=basic_agent(
            tools=[addition()],
            message_limit=50,
        ),
        scorer=includes(),
    )

    model = get_model("mockllm/model")
    log = eval(task, model=model)[0]
    assert log.status == "success"
    model_events = sum(
        1 for event in log.samples[0].transcript.events if event.event == "model"
    )
    assert model_events == 2


def test_setting_on_turn_injects_message():
    """Test that on_turn returning a string injects a user message."""
    call_count = 0

    async def inject_then_stop() -> bool | str | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Please try a different approach."
        return False

    s = Setting(on_turn=inject_then_stop)
    task = Task(
        dataset=[Sample(input="What is 1 + 1?", target="2")],
        setting=s,
        solver=basic_agent(
            tools=[addition()],
            message_limit=50,
        ),
        scorer=includes(),
    )

    model = get_model("mockllm/model")
    log = eval(task, model=model)[0]
    assert log.status == "success"

    user_messages = [
        m.content for m in log.samples[0].messages if isinstance(m, ChatMessageUser)
    ]
    assert "Please try a different approach." in user_messages


def test_factory_setting_per_sample():
    """Test that callable setting creates per-sample settings."""

    def make_setting(sample: Sample) -> Setting:
        tools_list: tuple[Tool, ...] = ()
        if sample.metadata and sample.metadata.get("needs_addition"):
            tools_list = (addition(),)
        return Setting(tools=tools_list)

    task = Task(
        dataset=[
            Sample(
                id="with_tool",
                input="What is 1 + 1?",
                target="2",
                metadata={"needs_addition": True},
            ),
            Sample(
                id="without_tool",
                input="Say hello",
                target="hello",
                metadata={"needs_addition": False},
            ),
        ],
        setting=make_setting,
        solver=basic_agent(
            tools=[],
            message_limit=5,
        ),
        scorer=includes(),
    )

    submit_output = ModelOutput.for_tool_call(
        model="mockllm/model",
        tool_name="submit",
        tool_arguments={"answer": "2"},
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=[submit_output] * 2,
    )

    log = eval(task, model=model)[0]
    assert log.status == "success"

    sample_with = next(s for s in log.samples if s.id == "with_tool")
    model_event = next(
        event for event in sample_with.transcript.events if event.event == "model"
    )
    tool_names = {t.name for t in model_event.tools}
    assert "addition" in tool_names

    sample_without = next(s for s in log.samples if s.id == "without_tool")
    model_event = next(
        event for event in sample_without.transcript.events if event.event == "model"
    )
    tool_names = {t.name for t in model_event.tools}
    assert "addition" not in tool_names
```

**Step 3: Run all tests**

Run: `uv run pytest tests/solver/test_setting.py tests/solver/test_basic_agent.py -v`
Expected: PASS (all tests)

**Step 4: Commit**

```bash
git add src/inspect_ai/__init__.py tests/solver/test_setting.py
git commit -m "feat: update top-level exports and add integration tests for Setting"
```

---

### Task 7: Remove old SampleContext files and references

**Files:**
- Delete: `src/inspect_ai/solver/_sample_context.py`
- Delete: `tests/solver/test_sample_context.py`

**Step 1: Delete old files**

```bash
rm src/inspect_ai/solver/_sample_context.py
rm tests/solver/test_sample_context.py
```

**Step 2: Verify no remaining references**

Run: `grep -r "sample_context\|SampleContext\|SandboxRole" src/inspect_ai/ tests/ --include="*.py" -l`
Expected: No files listed (all references have been replaced)

**Step 3: Run full verification**

Run: `uv run ruff check --fix && uv run ruff format src/ tests/`
Run: `uv run mypy --exclude tests/test_package src/inspect_ai/solver/_setting.py src/inspect_ai/solver/_task_state.py src/inspect_ai/_eval/task/task.py src/inspect_ai/_eval/task/run.py src/inspect_ai/agent/_react.py src/inspect_ai/solver/_basic_agent.py src/inspect_ai/solver/__init__.py src/inspect_ai/__init__.py`
Run: `uv run pytest tests/solver/test_setting.py tests/solver/test_basic_agent.py -v`

Expected: All clean, all pass.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old SampleContext/SandboxRole files"
```
