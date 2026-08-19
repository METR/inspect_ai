"""Prior-usage offsets on the sample timing contextvar."""

from __future__ import annotations

import time

from inspect_ai._util.working import (
    init_sample_working_time,
    report_sample_waiting_time,
    sample_total_time,
    sample_working_time,
)


def test_defaults_to_no_offset() -> None:
    init_sample_working_time(time.monotonic())

    assert sample_total_time() < 1.0
    assert sample_working_time() < 1.0


def test_prior_time_offsets_total() -> None:
    init_sample_working_time(time.monotonic(), prior_time=100.0)

    assert 100.0 <= sample_total_time() < 101.0


def test_prior_working_time_offsets_working() -> None:
    init_sample_working_time(time.monotonic(), prior_working_time=50.0)

    assert 50.0 <= sample_working_time() < 51.0


def test_offsets_are_independent() -> None:
    init_sample_working_time(
        time.monotonic(), prior_time=100.0, prior_working_time=50.0
    )
    report_sample_waiting_time(10.0)

    assert 100.0 <= sample_total_time() < 101.0
    # working excludes the 10s wait and adds only its own offset
    assert 40.0 <= sample_working_time() < 41.0
