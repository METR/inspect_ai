"""Tests for message pool deduplication in .eval files."""

from inspect_ai.event._model import ModelEvent
from inspect_ai.log._condense import condense_sample, resolve_sample_attachments
from inspect_ai.log._log import EvalSample
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
