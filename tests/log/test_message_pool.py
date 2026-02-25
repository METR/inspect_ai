"""Tests for message pool deduplication in .eval files."""

from inspect_ai.event._model import ModelEvent
from inspect_ai.log._log import EvalSample
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageUser
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
