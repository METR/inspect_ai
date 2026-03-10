# Backpressure-Only Server Buffering

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove circular buffer mode from the server, always use backpressure, and rely on the existing client-side `CircularByteBuffer` in `exec_remote_awaitable` for tail truncation.

**Architecture:** The server's `_OutputBuffer` drops circular mode and always applies backpressure (blocks reader when full). The `output_limit` parameter is removed from the entire server API chain. The client side is unchanged — `exec_remote_awaitable` already uses `CircularByteBuffer` to cap output.

**Tech Stack:** Python, asyncio, pydantic

---

## File Structure

**Server (inspect_sandbox_tools):**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_output_buffer.py` — remove circular mode
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_job.py` — remove `output_limit`, always backpressure
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_controller.py` — remove `output_limit` from `submit()`
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/json_rpc_methods.py` — stop passing `output_limit`
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/tool_types.py` — remove `output_limit` from `SubmitParams`

**Client (inspect_ai):**
- Modify: `src/inspect_ai/util/_sandbox/exec_remote.py` — remove `output_limit` from `_start()` and `exec_remote_streaming()`

**Tests:**
- Modify: `src/inspect_sandbox_tools/tests/test_output_buffer.py` — remove circular tests, update remaining tests
- Modify: `src/inspect_sandbox_tools/tests/test_exec_remote.py` — remove `TestOutputLimit` class

---

## Chunk 1: Server-side changes

### Task 1: Simplify `_OutputBuffer` to backpressure-only

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_output_buffer.py`

- [ ] **Step 1: Rewrite `_OutputBuffer` removing circular mode**

```python
"""Bounded output buffer with backpressure."""

import asyncio
from collections import deque


class _OutputBuffer:
    """Buffer for subprocess output with backpressure.

    Accumulates data up to max_bytes, then signals full. Caller must
    await wait_for_space() before writing more. When the buffer is full
    the reader task suspends, the kernel pipe fills, and the subprocess
    blocks on write — applying backpressure all the way to the source.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._chunks: deque[bytes] = deque()
        self._total_bytes = 0
        self._has_space = asyncio.Event()
        self._has_space.set()

    def write(self, data: bytes) -> None:
        """Append data to the buffer."""
        if not data:
            return
        self._chunks.append(data)
        self._total_bytes += len(data)
        if self._total_bytes >= self._max_bytes:
            self._has_space.clear()

    async def wait_for_space(self) -> None:
        """Block until buffer has space."""
        await self._has_space.wait()

    def drain(self) -> str:
        """Return all buffered data as a string and clear the buffer."""
        if not self._chunks:
            return ""
        result = b"".join(self._chunks).decode("utf-8", errors="replace")
        self._chunks.clear()
        self._total_bytes = 0
        self._has_space.set()
        return result

    def unblock(self) -> None:
        """Manually signal that space is available (e.g. on process exit)."""
        self._has_space.set()
```

- [ ] **Step 2: Verify file saved correctly**

Run: `uv run python -c "from inspect_sandbox_tools._remote_tools._exec_remote._output_buffer import _OutputBuffer; print('OK')"`

### Task 2: Remove `output_limit` from `Job`

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_job.py`

- [ ] **Step 1: Remove `output_limit` parameter from `Job.create()` and `__init__`**

In `create()`: remove the `output_limit` parameter and stop passing it to `cls()`.
In `__init__()`: remove the `output_limit` parameter. Always create backpressure buffers:

```python
def __init__(self, process: AsyncIOProcess) -> None:
    self._process = process
    self._stdout_buffer = _OutputBuffer(_BACKPRESSURE_BUFFER_SIZE)
    self._stderr_buffer = _OutputBuffer(_BACKPRESSURE_BUFFER_SIZE)
    ...
```

The `create()` call becomes `job = cls(process)`.

### Task 3: Remove `output_limit` from Controller, RPC, and types

**Files:**
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_controller.py`
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/json_rpc_methods.py`
- Modify: `src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/tool_types.py`

- [ ] **Step 1: Remove `output_limit` from `Controller.submit()`**

Remove the parameter and stop passing it to `Job.create()`.

- [ ] **Step 2: Remove `output_limit` from `exec_remote_start` RPC handler**

Stop passing `output_limit=params.output_limit` to `controller.submit()`.

- [ ] **Step 3: Remove `output_limit` from `SubmitParams`**

Delete the `output_limit: int | None = None` field and its docstring.

### Task 4: Remove `output_limit` from client-side `exec_remote.py`

**Files:**
- Modify: `src/inspect_ai/util/_sandbox/exec_remote.py`

- [ ] **Step 1: Remove `output_limit` from `_start()`**

Remove the parameter and stop adding it to `params`.

- [ ] **Step 2: Remove `output_limit` from `exec_remote_streaming()`**

Remove the parameter and stop passing it to `_start()`.

- [ ] **Step 3: Update `exec_remote_awaitable()` call**

Remove `output_limit=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE` from the `exec_remote_streaming()` call. The client-side `CircularByteBuffer` at lines 587-588 already handles truncation.

- [ ] **Step 4: Run linting**

Run: `uv run ruff check --fix src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/ src/inspect_ai/util/_sandbox/exec_remote.py`
Run: `uv run ruff format src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/ src/inspect_ai/util/_sandbox/exec_remote.py`

## Chunk 2: Test updates and commit

### Task 5: Update `test_output_buffer.py`

**Files:**
- Modify: `src/inspect_sandbox_tools/tests/test_output_buffer.py`

- [ ] **Step 1: Rewrite tests — remove circular tests, update backpressure tests**

Remove all circular-mode tests (`test_circular_*`) and the `test_circular_wait_for_space_returns_immediately` test.

Update remaining backpressure tests to use `_OutputBuffer(max_bytes=N)` (no `circular` param).

Update `test_empty_write_is_noop` to use the new constructor.

### Task 6: Remove `TestOutputLimit` from `test_exec_remote.py`

**Files:**
- Modify: `src/inspect_sandbox_tools/tests/test_exec_remote.py`

- [ ] **Step 1: Delete the entire `TestOutputLimit` class (lines 1066-1173)**

These tests tested server-side circular truncation via the `output_limit` RPC param, which no longer exists. The `output_limit` param would now be rejected by pydantic's `extra="forbid"`.

### Task 7: Run tests and commit

- [ ] **Step 1: Run unit tests**

Run: `cd src/inspect_sandbox_tools && uv run pytest tests/test_output_buffer.py -v`

- [ ] **Step 2: Run exec_remote tests (excluding slow integration tests)**

Run: `cd src/inspect_sandbox_tools && uv run pytest tests/test_exec_remote.py -v -k "not slow"`

- [ ] **Step 3: Lint everything**

Run: `uv run ruff check --fix && uv run ruff format`

- [ ] **Step 4: Commit**

```bash
git add src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_output_buffer.py \
  src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_job.py \
  src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/_controller.py \
  src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/json_rpc_methods.py \
  src/inspect_sandbox_tools/src/inspect_sandbox_tools/_remote_tools/_exec_remote/tool_types.py \
  src/inspect_ai/util/_sandbox/exec_remote.py \
  src/inspect_sandbox_tools/tests/test_output_buffer.py \
  src/inspect_sandbox_tools/tests/test_exec_remote.py
git commit -m "refactor: use backpressure-only on server, client-side circular buffer for truncation"
```
