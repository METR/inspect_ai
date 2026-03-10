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
