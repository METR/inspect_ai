import anyio
import pytest

from inspect_ai.util._limit import (
    LimitExceededError,
    message_limit,
    time_limit,
    token_limit,
)
from inspect_ai.util._limit_overrides import (
    sample_limit_override_scope,
    set_sample_limit_override,
)


def test_validates_limit_parameter() -> None:
    with pytest.raises(ValueError):
        time_limit(-0.1)


@pytest.mark.anyio
async def test_can_create_with_none_limit() -> None:
    with time_limit(None):
        pass


@pytest.mark.anyio
async def test_can_create_with_zero_limit() -> None:
    with pytest.raises(LimitExceededError):
        with time_limit(0):
            await anyio.sleep(0.1)


@pytest.mark.anyio
async def test_does_not_raise_error_when_limit_not_exceeded() -> None:
    with time_limit(10):
        pass


@pytest.mark.anyio
async def test_raises_error_when_limit_exceeded() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(0.1) as limit:
            await anyio.sleep(0.5)

    assert exc_info.value.type == "time"
    assert 0.0 < exc_info.value.value < 1.0  # approx. 0.1
    assert exc_info.value.limit == 0.1
    assert exc_info.value.source is limit


@pytest.mark.anyio
async def test_out_of_scope_limits_are_not_checked() -> None:
    with time_limit(0.1):
        pass

    await anyio.sleep(0.5)


@pytest.mark.anyio
async def test_outer_limits_are_enforced() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(0.1):
            with time_limit(10):
                await anyio.sleep(1)

    assert exc_info.value.limit == 0.1


@pytest.mark.anyio
async def test_inner_limits_are_enforced() -> None:
    with pytest.raises(LimitExceededError) as exc_info:
        with time_limit(10):
            with time_limit(0.1):
                await anyio.sleep(1)

    assert exc_info.value.limit == 0.1


def test_can_get_limit_value() -> None:
    limit = time_limit(10)

    assert limit.limit == 10


async def test_can_get_usage_while_context_manager_open() -> None:
    with time_limit(10) as limit:
        await anyio.sleep(0.1)

        assert 0.05 < limit.usage < 0.5  # approx. 0.1


async def test_can_get_usage_before_context_manager_opened() -> None:
    limit = time_limit(10)

    assert limit.usage == 0


async def test_can_get_usage_after_context_manager_closed() -> None:
    with time_limit(10) as limit:
        await anyio.sleep(0.1)

    await anyio.sleep(1)

    assert 0.05 < limit.usage < 0.5  # approx. 0.1


async def test_can_get_usage_nested() -> None:
    with time_limit(10) as outer_limit:
        await anyio.sleep(0.1)
        with time_limit(10) as inner_limit:
            await anyio.sleep(0.1)

    assert 0.15 < outer_limit.usage < 0.6  # approx. 0.2
    assert 0.05 < inner_limit.usage < 0.5  # approx. 0.1
    assert outer_limit.usage > inner_limit.usage


async def test_can_get_usage_after_limit_error() -> None:
    with pytest.raises(LimitExceededError):
        with time_limit(0.1) as limit:
            await anyio.sleep(0.5)

    assert 0.05 < limit.usage < 1.0  # approx. 0.1


async def test_can_get_remaining() -> None:
    limit = time_limit(10)
    with limit:
        assert limit.remaining is not None
        assert limit.remaining >= 9


@pytest.mark.anyio
async def test_does_not_mask_exception_raised_after_deadline() -> None:
    # If the deadline fires AND the body subsequently raises a non-Cancelled
    # exception (e.g. cleanup in a `finally` crashes), the original exception
    # must propagate rather than being masked by LimitExceededError.
    with pytest.raises(RuntimeError, match="cleanup crashed"):
        with time_limit(0.05):
            try:
                await anyio.sleep(1.0)
            finally:
                raise RuntimeError("cleanup crashed")


@pytest.mark.anyio
async def test_cannot_reuse_context_manager() -> None:
    limit = time_limit(10)
    with limit:
        pass

    with pytest.raises(RuntimeError) as exc_info:
        # Reusing the same Limit instance.
        with limit:
            pass

    assert "Each Limit may only be used once in a single 'with' block" in str(
        exc_info.value
    )


@pytest.mark.anyio
async def test_cannot_reuse_context_manager_in_stack() -> None:
    limit = time_limit(10)

    with pytest.raises(RuntimeError) as exc_info:
        with limit:
            # Reusing the same Limit instance in a stack.
            with limit:
                pass

    assert "Each Limit may only be used once in a single 'with' block" in str(
        exc_info.value
    )


@pytest.mark.anyio
async def test_seeded_time_shortens_the_deadline() -> None:
    limit = time_limit(1.0)
    limit._seed_usage(0.9)

    with pytest.raises(LimitExceededError) as exc_info:
        with limit:
            await anyio.sleep(0.5)

    assert exc_info.value.limit == 1.0
    assert exc_info.value.value > 0.9


@pytest.mark.anyio
async def test_seeded_time_reports_cumulative_usage() -> None:
    limit = time_limit(10)
    limit._seed_usage(5.0)

    with limit:
        assert 5.0 <= limit.usage < 6.0


@pytest.mark.parametrize("seed", [1.0, 1.5])
@pytest.mark.anyio
async def test_seeded_time_at_or_over_limit_cancels_immediately(seed: float) -> None:
    """A seed >= the limit clamps the remaining budget to 0, not negative."""
    limit = time_limit(1.0)
    limit._seed_usage(seed)

    with pytest.raises(LimitExceededError) as exc_info:
        with limit:
            await anyio.sleep(0.5)

    assert exc_info.value.limit == 1.0


@pytest.mark.anyio
async def test_seeded_time_override_refresh_keeps_the_seed() -> None:
    """A live override on a seeded scope must not hand back a fresh full budget.

    Regression guard: `_refresh_deadline` re-derives the deadline from
    `_remaining_limit()`, which subtracts the seed — a naive refresh keyed
    only off the new override value would reopen the full override amount.
    """
    limit = time_limit(10.0)
    limit._seed_usage(3.0)

    with sample_limit_override_scope(
        "seeded-time-refresh",
        time=limit,
        token=token_limit(None),
        message=message_limit(None),
    ):
        with limit:
            assert limit._start_time is not None
            assert limit._cancel_scope.deadline == pytest.approx(
                limit._start_time + 7.0
            )

            set_sample_limit_override("seeded-time-refresh", "time_limit", 5)
            # a fresh (unseeded) budget would put the deadline at start + 5;
            # the seed must still shorten it to start + (5 - 3)
            assert limit._cancel_scope.deadline == pytest.approx(
                limit._start_time + 2.0
            )

            set_sample_limit_override("seeded-time-refresh", "time_limit", None)


async def test_seed_usage_after_enter_raises() -> None:
    limit = time_limit(10)

    with limit:
        with pytest.raises(RuntimeError, match="before entering"):
            limit._seed_usage(1.0)
