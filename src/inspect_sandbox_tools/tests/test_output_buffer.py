"""Unit tests for _OutputBuffer."""

import asyncio

import pytest
from inspect_sandbox_tools._remote_tools._exec_remote._output_buffer import (
    _OutputBuffer,
)


def test_small_output_no_blocking():
    buf = _OutputBuffer(max_bytes=100)
    buf.write(b"hello")
    assert buf.drain() == "hello"


@pytest.mark.asyncio
async def test_blocks_when_full():
    buf = _OutputBuffer(max_bytes=5)
    buf.write(b"12345")
    # Buffer is full, wait_for_space should block
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(buf.wait_for_space(), timeout=0.05)


@pytest.mark.asyncio
async def test_unblocks_after_drain():
    buf = _OutputBuffer(max_bytes=5)
    buf.write(b"12345")
    # Buffer full - drain should free space
    buf.drain()
    # Should not block now
    await asyncio.wait_for(buf.wait_for_space(), timeout=0.05)


@pytest.mark.asyncio
async def test_unblock_wakes_waiter():
    buf = _OutputBuffer(max_bytes=5)
    buf.write(b"12345")

    woke = False

    async def waiter():
        nonlocal woke
        await buf.wait_for_space()
        woke = True

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    assert not woke
    buf.unblock()
    await asyncio.wait_for(task, timeout=0.1)
    assert woke


def test_preserves_all_data_under_limit():
    buf = _OutputBuffer(max_bytes=100)
    buf.write(b"aaa")
    buf.write(b"bbb")
    buf.write(b"ccc")
    assert buf.drain() == "aaabbbccc"


def test_drain_clears_buffer():
    buf = _OutputBuffer(max_bytes=100)
    buf.write(b"hello")
    buf.drain()
    assert buf.drain() == ""


def test_utf8_decode_with_replacement():
    buf = _OutputBuffer(max_bytes=100)
    buf.write(b"hello\xff\xfeworld")
    result = buf.drain()
    assert "hello" in result
    assert "world" in result
    assert "\ufffd" in result  # replacement character


def test_empty_write_is_noop():
    buf = _OutputBuffer(max_bytes=10)
    buf.write(b"")
    assert buf.drain() == ""
