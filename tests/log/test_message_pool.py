"""Tests for message pool deduplication in .eval files."""

import os
import tempfile

import pytest

from inspect_ai._util.constants import LOG_SCHEMA_VERSION
from inspect_ai.event._model import ModelEvent
from inspect_ai.event._subtask import SubtaskEvent
from inspect_ai.event._tool import ToolEvent
from inspect_ai.log._condense import (
    condense_sample,
    resolve_sample_attachments,
    resolve_sample_message_pool,
)
from inspect_ai.log._file import read_eval_log, write_eval_log
from inspect_ai.log._log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model._chat_message import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import ModelOutput


def test_model_event_input_refs_field_exists():
    """ModelEvent should accept an optional input_refs field."""
    event = ModelEvent(
        model="test-model",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
        input_refs=["msg1", "msg2"],
    )
    assert event.input_refs == ["msg1", "msg2"]


def test_model_event_input_refs_defaults_none():
    """input_refs should default to None."""
    event = ModelEvent(
        model="test-model",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    assert event.input_refs is None


def test_eval_sample_message_pool_field():
    """EvalSample should accept an optional message_pool dict."""
    pool = {
        "msg1": ChatMessageUser(content="Hello"),
        "msg2": ChatMessageAssistant(content="Hi there"),
    }
    sample = EvalSample(
        id="test",
        epoch=1,
        input="test input",
        target="test target",
        message_pool=pool,
    )
    assert sample.message_pool == pool
    assert sample.message_pool["msg1"].content == "Hello"


def test_eval_sample_message_pool_defaults_empty():
    """message_pool should default to an empty dict."""
    sample = EvalSample(
        id="test",
        epoch=1,
        input="test input",
        target="test target",
    )
    assert sample.message_pool == {}


def _make_sample_with_repeated_inputs() -> EvalSample:
    """Create a sample where model events have overlapping input messages."""
    msg_sys = ChatMessageSystem(content="You are helpful.")
    msg_user = ChatMessageUser(content="What is 2+2?")
    msg_asst = ChatMessageAssistant(content="4")
    msg_user2 = ChatMessageUser(content="And 3+3?")
    msg_asst2 = ChatMessageAssistant(content="6")

    event1 = ModelEvent(
        model="test-model",
        input=[msg_sys, msg_user],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    event2 = ModelEvent(
        model="test-model",
        input=[msg_sys, msg_user, msg_asst, msg_user2],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    event3 = ModelEvent(
        model="test-model",
        input=[msg_sys, msg_user, msg_asst, msg_user2, msg_asst2],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )

    return EvalSample(
        id="test",
        epoch=1,
        input="test",
        target="test",
        messages=[msg_sys, msg_user, msg_asst, msg_user2, msg_asst2],
        events=[event1, event2, event3],
    )


def test_condense_builds_message_pool():
    """condense_sample should extract messages into message_pool."""
    sample = _make_sample_with_repeated_inputs()
    condensed = condense_sample(sample)

    # message_pool should contain all unique messages
    assert len(condensed.message_pool) > 0

    # Each model event should have input_refs
    model_events = [e for e in condensed.events if isinstance(e, ModelEvent)]
    for event in model_events:
        assert event.input_refs is not None
        assert len(event.input) == 0
        assert all(ref in condensed.message_pool for ref in event.input_refs)


def test_condense_message_pool_no_duplication():
    """Messages appearing in multiple events should be stored once."""
    sample = _make_sample_with_repeated_inputs()
    condensed = condense_sample(sample)

    # 5 unique messages across 3 events
    assert len(condensed.message_pool) == 5

    # Total refs across events should be 2 + 4 + 5 = 11
    total_refs = sum(
        len(e.input_refs)
        for e in condensed.events
        if isinstance(e, ModelEvent) and e.input_refs is not None
    )
    assert total_refs == 11


def test_resolve_reconstructs_model_event_inputs():
    """resolve_sample_attachments should rebuild input from input_refs + message_pool."""
    sample = _make_sample_with_repeated_inputs()
    condensed = condense_sample(sample)

    # Resolve
    resolved = resolve_sample_attachments(condensed, "full")

    # Model events should have full input restored
    model_events = [e for e in resolved.events if isinstance(e, ModelEvent)]
    assert len(model_events[0].input) == 2
    assert len(model_events[1].input) == 4
    assert len(model_events[2].input) == 5

    # input_refs should be cleared
    for event in model_events:
        assert event.input_refs is None

    # message_pool should be cleared
    assert resolved.message_pool == {}


def test_condense_resolve_round_trip():
    """Condense then resolve should preserve message content."""
    sample = _make_sample_with_repeated_inputs()
    condensed = condense_sample(sample)
    resolved = resolve_sample_attachments(condensed, "full")

    original_model_events = [e for e in sample.events if isinstance(e, ModelEvent)]
    resolved_model_events = [e for e in resolved.events if isinstance(e, ModelEvent)]

    for orig, res in zip(original_model_events, resolved_model_events):
        assert len(orig.input) == len(res.input)
        for orig_msg, res_msg in zip(orig.input, res.input):
            assert orig_msg.role == res_msg.role
            # Content may have been through attachment condensation (strings > 100 chars),
            # but our short test messages should survive unchanged
            assert orig_msg.content == res_msg.content


def test_schema_version_is_3():
    """Schema version should be bumped to 3 for message pool support."""
    assert LOG_SCHEMA_VERSION == 3


def test_read_v2_eval_file():
    """Reading a v2 .eval file should work -- empty message_pool, input populated."""
    log_file = os.path.join("tests", "log", "test_eval_log", "log_read_sample.eval")
    if not os.path.exists(log_file):
        pytest.skip("Test fixture not available")

    log = read_eval_log(log_file)
    # v2 file should still read; version stays as written in the file
    assert log.version >= 2
    if log.samples:
        sample = log.samples[0]
        assert sample.message_pool == {}
        model_events = [e for e in sample.events if isinstance(e, ModelEvent)]
        for event in model_events:
            assert event.input_refs is None


def _make_eval_log_with_model_events() -> EvalLog:
    """Create a minimal EvalLog with model events that have repeated inputs.

    The samples are pre-condensed (condense_sample called) since write_eval_log
    does not call condense_sample itself -- that happens at eval run time.
    """
    sample = _make_sample_with_repeated_inputs()
    condensed_sample = condense_sample(sample)
    return EvalLog(
        version=LOG_SCHEMA_VERSION,
        status="success",
        eval=EvalSpec(
            task="test_task",
            task_version=0,
            task_id="test",
            model="test-model",
            dataset=EvalDataset(name="test", samples=1),
            config=EvalConfig(),
            created="2025-01-01T00:00:00Z",
        ),
        plan=EvalPlan(),
        results=EvalResults(total_samples=1, completed_samples=1),
        stats=EvalStats(
            started_at="2025-01-01T00:00:00Z",
            completed_at="2025-01-01T00:01:00Z",
        ),
        samples=[condensed_sample],
    )


def test_write_read_round_trip_eval_format():
    """Write a v3 .eval file and read it back -- message pool is always resolved on read."""
    log = _make_eval_log_with_model_events()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.eval")
        write_eval_log(log, path)

        read_log = read_eval_log(path)
        assert read_log.version == 3
        assert read_log.samples is not None
        assert len(read_log.samples) == 1

        sample = read_log.samples[0]
        # message_pool is always resolved on read, so it should be empty
        assert sample.message_pool == {}

        # Model events should have full input restored (not input_refs)
        model_events = [e for e in sample.events if isinstance(e, ModelEvent)]
        assert len(model_events[0].input) == 2
        assert len(model_events[1].input) == 4
        assert len(model_events[2].input) == 5
        for event in model_events:
            assert event.input_refs is None


def test_write_read_round_trip_with_resolve():
    """Write v3, read with resolve_attachments -- should get full inputs back."""
    log = _make_eval_log_with_model_events()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.eval")
        write_eval_log(log, path)

        read_log = read_eval_log(path, resolve_attachments="full")
        assert read_log.samples is not None
        sample = read_log.samples[0]

        # After resolution, message_pool should be empty
        assert sample.message_pool == {}

        # Model events should have full input reconstructed
        model_events = [e for e in sample.events if isinstance(e, ModelEvent)]
        assert len(model_events[0].input) == 2  # sys + user
        assert len(model_events[1].input) == 4  # sys + user + asst + user2
        assert len(model_events[2].input) == 5  # all 5

        # input_refs should be cleared
        for event in model_events:
            assert event.input_refs is None


def test_read_without_resolve_still_resolves_message_pool():
    """Reading with resolve_attachments=False should still resolve message pool."""
    log = _make_eval_log_with_model_events()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.eval")
        write_eval_log(log, path)

        # Read WITHOUT resolve_attachments
        read_log = read_eval_log(path, resolve_attachments=False)
        assert read_log.samples is not None
        sample = read_log.samples[0]

        # Message pool should still be resolved (cleared)
        assert sample.message_pool == {}

        # Model events should have full input populated
        model_events = [e for e in sample.events if isinstance(e, ModelEvent)]
        assert len(model_events[0].input) == 2
        assert len(model_events[1].input) == 4
        assert len(model_events[2].input) == 5
        for event in model_events:
            assert event.input_refs is None


def test_condense_anonymous_message_ids():
    """Messages without IDs should get _anon_ hash-based keys."""
    msg1 = ChatMessageUser(content="Hello")
    msg1.id = None  # explicitly no ID
    msg2 = ChatMessageAssistant(content="Hi")
    msg2.id = None

    event = ModelEvent(
        model="test-model",
        input=[msg1, msg2],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    sample = EvalSample(id="test", epoch=1, input="test", target="test", events=[event])
    condensed = condense_sample(sample)

    # All keys should start with _anon_
    for key in condensed.message_pool:
        assert key.startswith("_anon_"), f"Expected _anon_ prefix, got: {key}"
    assert len(condensed.message_pool) == 2


def test_resolve_missing_refs_drops_message():
    """Missing refs should be dropped from input (only valid refs resolved)."""
    sample = EvalSample(
        id="test",
        epoch=1,
        input="test",
        target="test",
        message_pool={"msg1": ChatMessageUser(content="Hello")},
        events=[
            ModelEvent(
                model="test-model",
                input=[],
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
                output=ModelOutput(),
                input_refs=["msg1", "nonexistent_ref"],
            )
        ],
    )
    resolved = resolve_sample_message_pool(sample)

    # Should have resolved the valid ref and dropped the missing one
    model_events = [e for e in resolved.events if isinstance(e, ModelEvent)]
    assert len(model_events[0].input) == 1
    assert model_events[0].input[0].content == "Hello"
    assert model_events[0].input_refs is None


def test_condense_nested_events_in_tool_event():
    """ModelEvents nested inside ToolEvent.events should be condensed."""
    msg_sys = ChatMessageSystem(content="System prompt")
    msg_user = ChatMessageUser(content="Question")

    nested_model_event = ModelEvent(
        model="test-model",
        input=[msg_sys, msg_user],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    tool_event = ToolEvent(
        id="tool-1",
        function="my_tool",
        arguments={},
        events=[nested_model_event],
    )
    # Top-level model event shares messages
    top_model_event = ModelEvent(
        model="test-model",
        input=[msg_sys, msg_user],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
    )
    sample = EvalSample(
        id="test",
        epoch=1,
        input="test",
        target="test",
        events=[top_model_event, tool_event],
    )
    condensed = condense_sample(sample)

    # Both top-level and nested model events should have input_refs
    top_events = [e for e in condensed.events if isinstance(e, ModelEvent)]
    assert len(top_events) == 1
    assert top_events[0].input_refs is not None

    tool_events = [e for e in condensed.events if isinstance(e, ToolEvent)]
    nested_model = [e for e in tool_events[0].events if isinstance(e, ModelEvent)]
    assert len(nested_model) == 1
    assert nested_model[0].input_refs is not None

    # Messages shared between top and nested should be deduped
    assert len(condensed.message_pool) == 2  # msg_sys + msg_user


def test_resolve_nested_events_in_subtask_event():
    """ModelEvents nested inside SubtaskEvent.events should be resolved."""
    msg = ChatMessageUser(content="Hello")
    nested_model_event = ModelEvent(
        model="test-model",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(),
        input_refs=["ref1"],
    )
    subtask_event = SubtaskEvent(
        name="my_subtask",
        input={},
        events=[nested_model_event],
    )
    sample = EvalSample(
        id="test",
        epoch=1,
        input="test",
        target="test",
        message_pool={"ref1": msg},
        events=[subtask_event],
    )
    resolved = resolve_sample_message_pool(sample)

    subtask_events = [e for e in resolved.events if isinstance(e, SubtaskEvent)]
    nested_model = [e for e in subtask_events[0].events if isinstance(e, ModelEvent)]
    assert len(nested_model[0].input) == 1
    assert nested_model[0].input[0].content == "Hello"
    assert nested_model[0].input_refs is None
