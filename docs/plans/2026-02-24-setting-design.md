# Design: Setting (replaces SampleContext)

## Problem

Inspect AI couples tools to solvers and sandboxes to samples. This causes problems when:

- Some samples need specific tools (custom CLIs, binaries)
- Some samples need per-turn callbacks (check completion, advance time, logging)
- Scaffolding needs to know which sandboxes the agent should work in and what user to use
- Some sandboxes (targets, resources) should be hidden from the agent entirely

The initial `SampleContext` implementation addressed tools and callbacks. This design replaces it with `Setting` — a more opinionated model that also handles workspace semantics.

## Data Model

### Workspace

```python
class Workspace(NamedTuple):
    """A sandbox environment the agent should work in."""

    sandbox: str = "default"
    """Sandbox environment name (matches docker-compose service name)."""

    description: str = ""
    """Human-readable description of this workspace for the agent."""

    user: str | None = None
    """User to run commands as in this sandbox."""
```

Only agent-accessible sandboxes are listed as workspaces. Non-workspace sandboxes (targets, resources) are omitted — hidden from the agent. Scaffolding and scoring code can still access them via `sandbox("target")` directly.

### Setting

```python
OnTurn = Callable[[], Awaitable[bool | str | None]]

class Setting(NamedTuple):
    """Execution setting provided by the problem definition."""

    workspaces: tuple[Workspace, ...] = ()
    """Sandboxes the agent should work in. First is primary."""

    tools: tuple[Tool | ToolDef, ...] = ()
    """Custom tools the agent needs (CLIs, binaries, etc.).
    Standard tools (bash, editor) are created by scaffolding from workspace metadata."""

    on_turn: OnTurn | None = None
    """Callback fired each iteration of the agent loop.
    Returns False to stop, str to inject a user message, None/True to continue."""
```

### Accessor

```python
def setting() -> Setting | None:
    """Get the Setting for the current sample, if any."""
```

## Integration Points

### TaskState

- `_setting: Setting | None` stored on TaskState
- Property with getter and setter
- Not included in `state_jsonable()` — effects are logged, not the setting itself

### Task

```python
class Task:
    def __init__(
        self,
        ...,
        sandbox: SandboxEnvironmentType | None = None,
        setting: Setting | Callable[[Sample], Setting] | None = None,
        ...,
    ):
```

`Callable[[Sample], Setting]` supports per-sample configuration (different tools/workspaces per sample).

### Eval Loop

Applied after `deepcopy(TaskState(...))` in `resolve_dataset()` to avoid deepcopy issues with callables:

```python
if task_setting is not None:
    for state, sample in zip(states, samples):
        state.setting = task_setting(sample) if callable(task_setting) else task_setting
```

## Scaffolding Consumption

### Tool Merging

`Setting.tools` (custom tools) are prepended to solver tools. On name conflict, the setting's tool wins (problem definition takes precedence over scaffolding).

### Workspace Tool Creation

Scaffolding creates standard tools for each workspace using `ws.sandbox` and `ws.user`:

```python
s = setting()
if s is not None:
    for ws in s.workspaces:
        tools.append(bash(sandbox=ws.sandbox, user=ws.user))
```

The choice of which tools to create is scaffolding's decision. `react()` and `basic_agent()` start minimal (bash only). More opinionated scaffolding can create richer tool sets.

### on_turn Callback

Fires every iteration of the agent loop, after tool results but before `on_continue`:

- `False`: sets `TaskState.completed = True`, breaks
- `str`: appends as `ChatMessageUser`, skips `on_continue` for that iteration
- `None` or `True`: continues normally

## Usage Examples

### Simple: one workspace

```python
Task(
    sandbox=("docker", "compose.yaml"),
    setting=Setting(
        workspaces=(Workspace(description="Debian workspace"),),
    ),
)
```

### Custom tools + workspace

```python
Task(
    sandbox=("docker", "compose.yaml"),
    setting=Setting(
        workspaces=(
            Workspace(sandbox="default", description="Workspace with /opt/tools/decompile", user="hacker"),
        ),
        tools=(custom_decompile_tool(),),
    ),
)
```

### Multi-sandbox, hidden target

```python
Task(
    sandbox=("docker", "compose.yaml"),  # defines default + target services
    setting=Setting(
        workspaces=(
            Workspace(sandbox="default", description="Workspace. Target at port 8080.", user="hacker"),
            # "target" sandbox not listed — hidden from agent
        ),
    ),
)
```

### Per-sample factory

```python
Task(
    sandbox=("docker", "compose.yaml"),
    setting=lambda sample: Setting(
        workspaces=(Workspace(user=sample.metadata.get("user", "nobody")),),
        tools=tuple(sample.metadata.get("extra_tools", [])),
    ),
)
```

## Exports

- `inspect_ai.solver`: `Setting`, `Workspace`, `OnTurn`, `setting`
- `inspect_ai`: `Setting`, `Workspace`

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `Setting` not `SampleContext` | More opinionated, matches Control Arena's "Setting" concept |
| `Workspace` not `SandboxRole` | Only agent-accessible sandboxes are modeled; no role enum needed |
| `workspaces` is a tuple, not a dict | Order matters (first = primary), immutable, no key/value mismatch |
| No tools on Workspace | Scaffolding decides standard tools; custom tools go in `Setting.tools` |
| Non-workspace sandboxes are omitted | Hidden from agent. Scoring/callbacks access them via `sandbox()` directly |
| `setting` param after `sandbox` on Task | Both are execution environment concerns |
