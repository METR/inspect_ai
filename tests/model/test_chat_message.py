from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from inspect_ai._util.content import ContentImage, ContentText
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)


def test_chat_message_text_setter_ordering():
    # 1. Message with text first, then image
    msg1 = ChatMessageUser(
        content=[
            ContentText(text="initial text"),
            ContentImage(image="http://example.com/image.png"),
        ]
    )
    msg1.text = "updated text"

    # The text property setter should replace the first text item in-place
    assert isinstance(msg1.content, list)
    assert len(msg1.content) == 2
    assert isinstance(msg1.content[0], ContentText)
    assert msg1.content[0].text == "updated text"
    assert isinstance(msg1.content[1], ContentImage)

    # 2. Message with image first, then text
    msg2 = ChatMessageUser(
        content=[
            ContentImage(image="http://example.com/image.png"),
            ContentText(text="initial text"),
        ]
    )
    msg2.text = "updated text"

    # Should preserve text at index 1
    assert isinstance(msg2.content, list)
    assert len(msg2.content) == 2
    assert isinstance(msg2.content[0], ContentImage)
    assert isinstance(msg2.content[1], ContentText)
    assert msg2.content[1].text == "updated text"

    # 3. Message with multiple text items
    msg3 = ChatMessageUser(
        content=[
            ContentText(text="text 1"),
            ContentImage(image="http://example.com/image.png"),
            ContentText(text="text 2"),
        ]
    )
    msg3.text = "updated text"

    # Multiple text items should collapse into one at the initial text position
    assert isinstance(msg3.content, list)
    assert len(msg3.content) == 2
    assert isinstance(msg3.content[0], ContentText)
    assert msg3.content[0].text == "updated text"
    assert isinstance(msg3.content[1], ContentImage)

    # 4. Message with image only (no text item)
    msg4 = ChatMessageUser(
        content=[
            ContentImage(image="http://example.com/image.png"),
        ]
    )
    msg4.text = "inserted text"

    # Should append text after the existing content
    assert isinstance(msg4.content, list)
    assert len(msg4.content) == 2
    assert isinstance(msg4.content[0], ContentImage)
    assert isinstance(msg4.content[1], ContentText)
    assert msg4.content[1].text == "inserted text"


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"role": "system", "content": "x"}, ChatMessageSystem),
        ({"role": "user", "content": "x"}, ChatMessageUser),
        ({"role": "assistant", "content": "x"}, ChatMessageAssistant),
        ({"role": "tool", "content": "x", "tool_call_id": "a"}, ChatMessageTool),
        ({"content": "x"}, ChatMessageSystem),
        ({"content": "x", "tool_call_id": "a"}, ChatMessageTool),
        (ChatMessageUser(content="x"), ChatMessageUser),
    ],
)
def test_chat_message_validates_by_role(value: Any, expected: type) -> None:
    assert type(TypeAdapter(ChatMessage).validate_python(value)) is expected


@pytest.mark.parametrize(
    "value", [{"role": "bogus", "content": "x"}, {"role": "user"}, "user"]
)
def test_chat_message_rejects_invalid(value: Any) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ChatMessage).validate_python(value)


def test_chat_message_json_schema_is_plain_union() -> None:
    schema = TypeAdapter(ChatMessage).json_schema()
    assert "oneOf" not in schema
    assert [m["$ref"].rsplit("/", 1)[-1] for m in schema["anyOf"]] == [
        "ChatMessageSystem",
        "ChatMessageUser",
        "ChatMessageAssistant",
        "ChatMessageTool",
    ]


def _model_event(input: Any) -> Any:
    from inspect_ai.event._model import ModelEvent
    from inspect_ai.model import GenerateConfig, ModelOutput

    return ModelEvent(
        model="m",
        input=input,
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("m", ""),
    )


def test_model_event_input_keeps_message_instances_in_a_new_list():
    messages: list[ChatMessage] = [
        ChatMessageSystem(content="s"),
        ChatMessageUser(content="u"),
        ChatMessageAssistant(content="a"),
        ChatMessageTool(content="t", tool_call_id="1"),
    ]
    event = _model_event(messages)
    assert event.input is not messages
    assert all(x is y for x, y in zip(event.input, messages, strict=True))


def test_model_event_input_validates_dicts():
    event = _model_event(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    )
    assert [type(m) for m in event.input] == [ChatMessageUser, ChatMessageAssistant]
    with pytest.raises(ValidationError):
        _model_event([{"role": "user"}])
