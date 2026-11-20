"""The METR fork's eval-set protocol numbers (see `eval_set_fork_version`)."""

import pytest

from inspect_ai._eval.eval_set_fork_version import (
    fork_protocol_version,
    protocol_version_understood,
)
from inspect_ai._eval.eval_set_manifest import EVAL_SET_CAPTURE_VERSION
from inspect_ai._eval.eval_set_selection import EVAL_SET_SELECTION_VERSION


def test_a_fork_version_sits_above_upstreams_range_and_follows_its_version() -> None:
    assert fork_protocol_version(8, 1) == 9081
    assert fork_protocol_version(9, 1) == 9091
    assert fork_protocol_version(3, 2) == 9032


@pytest.mark.parametrize(("upstream", "position"), [(8, 0), (8, 10), (0, 1), (100, 1)])
def test_a_position_or_version_that_would_carry_into_another_digit_is_refused(
    upstream: int, position: int
) -> None:
    with pytest.raises(ValueError, match="No fork version"):
        fork_protocol_version(upstream, position)


@pytest.mark.parametrize(
    ("version", "understood"),
    [
        (1, True),
        (8, True),
        (9, False),
        (99, False),
        (9000, False),
        (9081, True),
        (9082, False),
        (9091, False),
    ],
)
def test_a_later_upstream_version_is_not_read_as_an_older_fork_one(
    version: int, understood: bool
) -> None:
    assert protocol_version_understood(version, upstream=8, current=9081) is understood


def test_the_versions_this_build_speaks() -> None:
    """A runner pins these exactly (Hawk: `hawk/core/sharding.py`), so a rebase that moves either must move the runner's with it."""
    assert EVAL_SET_SELECTION_VERSION == 9082
    assert EVAL_SET_CAPTURE_VERSION == 9032
