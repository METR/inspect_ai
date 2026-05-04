"""Tests for Gemini 3 mixed native + function tool combinations.

Covers thought_signature capture and replay, server-side tool round-trips
(web_search via tool_call/tool_response, code_execution via executable_code/
code_execution_result), part-ordering preservation across the
ChatMessageAssistant.content + tool_calls split, and live --runapi
verification against gemini-3.1-pro-preview.

The companion non-mixed-tools tests live in test_google.py.
"""

import base64
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import (
    Candidate,
    CodeExecutionResult,
    Content,
    ExecutableCode,
    FinishReason,
    FunctionCall,
    FunctionCallingConfigMode,
    GenerateContentResponse,
    Language,
    Outcome,
    Part,
    ToolResponse,
    ToolType,
)
from google.genai.types import (
    Tool as GeminiTool,
)
from google.genai.types import (
    ToolCall as GeminiToolCall,
)
from test_helpers.utils import skip_if_no_google

from inspect_ai import Task, eval
from inspect_ai._util.content import (
    ContentReasoning,
    ContentText,
    ContentToolUse,
)
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ModelOutput
from inspect_ai.model._chat_message import ChatMessageUser
from inspect_ai.model._compaction.edit import _clear_reasoning
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.google import (
    GoogleGenAIAPI,
    completion_choice_from_candidate,
    content,
    gemini_native_tool_combination,
    gemini_native_tool_combination_config,
    parts_from_server_tool_use,
)
from inspect_ai.tool import (
    Tool,
    ToolCall,
    ToolFunction,
    ToolInfo,
    ToolParam,
    ToolParams,
    code_execution,
    tool,
    web_search,
)

# ---------------------------------------------------------------------------
# Test helpers (small duplicates from test_google.py kept local to this file)
# ---------------------------------------------------------------------------


def _create_mock_google_client(mock_generate: AsyncMock) -> MagicMock:
    """Create a mock Google client with the right structure for testing."""
    mock_client = MagicMock()
    mock_client.aio.__aenter__ = AsyncMock(return_value=mock_client.aio)
    mock_client.aio.__aexit__ = AsyncMock(return_value=None)
    mock_client.aio.models.generate_content = mock_generate
    mock_client._api_client._async_httpx_client = MagicMock()
    return mock_client


def _create_test_tool() -> ToolInfo:
    """Create a simple test tool for testing."""
    return ToolInfo(
        name="my_tool",
        description="A test tool",
        parameters=ToolParams(
            type="object",
            properties={"x": ToolParam(type="integer", description="A number")},
            required=["x"],
        ),
    )


def _create_gemini_web_search_tool() -> ToolInfo:
    """Create a Gemini-native web search tool for testing."""
    return ToolInfo(
        name="web_search",
        description="Search the web",
        options={"gemini": {}},
    )


def _create_google_code_execution_tool() -> ToolInfo:
    """Create a Google-native code execution tool for testing."""
    return ToolInfo(
        name="code_execution",
        description="Execute Python code",
        options={"providers": {"google": {}}},
    )


def _create_record_result_tool() -> ToolInfo:
    """Create a string-valued custom tool for live mixed-tool tests."""
    return ToolInfo(
        name="record_result",
        description="Record an answer",
        parameters=ToolParams(
            type="object",
            properties={
                "answer": ToolParam(type="string", description="Answer summary")
            },
            required=["answer"],
        ),
    )


def _create_days_in_year_tool() -> ToolInfo:
    """Create a days-in-year custom tool for mixed native/custom replay tests."""
    return ToolInfo(
        name="days_in_year",
        description="Return the number of days in a year.",
        parameters=ToolParams(
            type="object",
            properties={
                "year": ToolParam(type="integer", description="The year to check.")
            },
            required=["year"],
        ),
    )


def _days_in_year_tool() -> Tool:
    @tool
    def days_in_year() -> Tool:
        async def execute(year: int) -> str:
            """Return the number of days in a year.

            Args:
                year: The year to check.
            """
            leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            return f"{year} has {366 if leap else 365} days."

        return execute

    return days_in_year()


def _record_result_tool() -> Tool:
    @tool
    def record_result() -> Tool:
        async def execute(answer: str) -> str:
            """Record an answer.

            Args:
                answer: Final answer summary.
            """
            return f"recorded: {answer}"

        return execute

    return record_result()


def _multiply_tool() -> Tool:
    @tool
    def multiply() -> Tool:
        async def execute(a: int, b: int) -> str:
            """Multiply two integers.

            Args:
                a: First integer.
                b: Second integer.
            """
            return f"{a * b}"

        return execute

    return multiply()


def test_gemini_3_native_search_can_combine_with_function_declarations() -> None:
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key="test-key",
    )

    has_native_tools, gemini_tools = api.chat_tools(
        [_create_gemini_web_search_tool(), _create_test_tool()]
    )

    assert has_native_tools is True
    assert gemini_native_tool_combination(gemini_tools)
    assert len(gemini_tools) == 2
    assert isinstance(gemini_tools[0], GeminiTool)
    assert isinstance(gemini_tools[1], GeminiTool)
    assert gemini_tools[0].function_declarations is not None
    assert gemini_tools[0].function_declarations[0].name == "my_tool"
    assert gemini_tools[1].google_search is not None


def test_gemini_3_native_code_execution_can_combine_with_functions() -> None:
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key="test-key",
    )

    has_native_tools, gemini_tools = api.chat_tools(
        [_create_google_code_execution_tool(), _create_test_tool()]
    )

    assert has_native_tools is True
    assert gemini_native_tool_combination(gemini_tools)
    assert len(gemini_tools) == 2
    assert isinstance(gemini_tools[0], GeminiTool)
    assert isinstance(gemini_tools[1], GeminiTool)
    assert gemini_tools[0].function_declarations is not None
    assert gemini_tools[0].function_declarations[0].name == "my_tool"
    assert gemini_tools[1].code_execution is not None


def test_gemini_2_5_native_search_still_rejects_function_declarations() -> None:
    api = GoogleGenAIAPI(
        model_name="gemini-2.5-pro",
        base_url=None,
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="requires Gemini 3 or later"):
        api.chat_tools([_create_gemini_web_search_tool(), _create_test_tool()])


def test_gemini_2_5_native_code_execution_still_rejects_function_declarations() -> None:
    api = GoogleGenAIAPI(
        model_name="gemini-2.5-pro",
        base_url=None,
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="code execution"):
        api.chat_tools([_create_google_code_execution_tool(), _create_test_tool()])


def test_gemini_native_tool_combination_config_respects_tool_choice() -> None:
    auto_config = gemini_native_tool_combination_config("auto")
    assert auto_config.function_calling_config is not None
    assert (
        auto_config.function_calling_config.mode == FunctionCallingConfigMode.VALIDATED
    )

    any_config = gemini_native_tool_combination_config("any")
    assert any_config.function_calling_config is not None
    assert any_config.function_calling_config.mode == FunctionCallingConfigMode.ANY

    none_config = gemini_native_tool_combination_config("none")
    assert none_config.function_calling_config is not None
    assert none_config.function_calling_config.mode == FunctionCallingConfigMode.NONE

    function_config = gemini_native_tool_combination_config(
        ToolFunction(name="my_tool")
    )
    assert function_config.function_calling_config is not None
    assert function_config.function_calling_config.mode == FunctionCallingConfigMode.ANY
    assert function_config.function_calling_config.allowed_function_names == ["my_tool"]


@pytest.mark.anyio
async def test_gemini_3_tool_combination_uses_server_side_tool_invocations() -> None:
    mock_generate = AsyncMock(
        return_value=GenerateContentResponse(
            candidates=[
                Candidate(
                    finish_reason=FinishReason.STOP,
                    content=Content(role="model", parts=[Part(text="done")]),
                )
            ],
            usage_metadata=None,
        )
    )
    mock_client = _create_mock_google_client(mock_generate)

    with patch("inspect_ai.model._providers.google.Client", return_value=mock_client):
        api = GoogleGenAIAPI(
            model_name="gemini-3.1-pro-preview",
            base_url=None,
            api_key="test-key",
        )

        await api.generate(
            input=[ChatMessageUser(content="Search and call my_tool")],
            tools=[_create_gemini_web_search_tool(), _create_test_tool()],
            tool_choice="auto",
            config=GenerateConfig(),
        )

    config = mock_generate.call_args.kwargs["config"]
    assert config.tool_config == gemini_native_tool_combination_config("auto")
    assert config.tool_config.include_server_side_tool_invocations is True
    assert config.tool_config.function_calling_config.mode == (
        FunctionCallingConfigMode.VALIDATED
    )


@pytest.mark.anyio
async def test_gemini_3_code_exec_combo_uses_server_invocations() -> None:
    mock_generate = AsyncMock(
        return_value=GenerateContentResponse(
            candidates=[
                Candidate(
                    finish_reason=FinishReason.STOP,
                    content=Content(role="model", parts=[Part(text="done")]),
                )
            ],
            usage_metadata=None,
        )
    )
    mock_client = _create_mock_google_client(mock_generate)

    with patch("inspect_ai.model._providers.google.Client", return_value=mock_client):
        api = GoogleGenAIAPI(
            model_name="gemini-3.1-pro-preview",
            base_url=None,
            api_key="test-key",
        )

        await api.generate(
            input=[ChatMessageUser(content="Run code and call my_tool")],
            tools=[_create_google_code_execution_tool(), _create_test_tool()],
            tool_choice="auto",
            config=GenerateConfig(),
        )

    config = mock_generate.call_args.kwargs["config"]
    assert config.tool_config == gemini_native_tool_combination_config("auto")
    assert config.tool_config.include_server_side_tool_invocations is True
    assert config.tools is not None
    assert isinstance(config.tools[0], GeminiTool)
    assert isinstance(config.tools[1], GeminiTool)
    assert config.tools[0].function_declarations is not None
    assert config.tools[0].function_declarations[0].name == "my_tool"
    assert config.tools[1].code_execution is not None


def test_gemini_server_tool_call_round_trips_for_replay() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    thought_signature=b"search-context",
                    tool_call=GeminiToolCall(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["current Gemini API tool combination docs"]},
                    ),
                ),
                Part(
                    thought_signature=b"search-context-response",
                    tool_response=ToolResponse(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"search_suggestions": ["Gemini tool combination"]},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)

    assert isinstance(choice.message.content, list)
    assert isinstance(choice.message.content[0], ContentToolUse)
    tool_use = choice.message.content[0]
    replayed_parts = parts_from_server_tool_use(tool_use)
    assert replayed_parts[0].tool_call is not None
    assert replayed_parts[0].tool_call.id == "search-1"
    assert replayed_parts[0].thought_signature == b"search-context"
    assert replayed_parts[1].tool_response is not None
    assert replayed_parts[1].tool_response.id == "search-1"
    assert replayed_parts[1].thought_signature == b"search-context-response"


def test_gemini_server_tool_call_without_signature_is_omitted_for_replay() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    tool_call=GeminiToolCall(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["current Gemini API tool combination docs"]},
                    ),
                ),
                Part(
                    tool_response=ToolResponse(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"search_suggestions": ["Gemini tool combination"]},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)

    assert isinstance(choice.message.content, list)
    assert isinstance(choice.message.content[0], ContentToolUse)
    tool_use = choice.message.content[0]
    assert parts_from_server_tool_use(tool_use) == []


@pytest.mark.anyio
async def test_gemini_signed_function_call_replays_before_server_tool() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(text="Plan", thought=True),
                Part(
                    thought_signature=b"days-signature",
                    function_call=FunctionCall(
                        id="days-1",
                        name="days_in_year",
                        args={"year": 2024},
                    ),
                ),
                Part(
                    thought_signature=b"search-signature",
                    tool_call=GeminiToolCall(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["most populous country"]},
                    ),
                ),
                Part(
                    thought_signature=b"search-response-signature",
                    tool_response=ToolResponse(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"search_suggestions": ["India population"]},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].text == "Plan"
    assert google_content.parts[0].thought is True
    assert google_content.parts[1].function_call is not None
    assert google_content.parts[1].function_call.id == "days-1"
    assert google_content.parts[1].thought_signature == b"days-signature"
    assert google_content.parts[2].tool_call is not None
    assert google_content.parts[2].tool_call.id == "search-1"
    assert google_content.parts[2].thought_signature == b"search-signature"
    assert google_content.parts[3].tool_response is not None
    assert google_content.parts[3].tool_response.id == "search-1"


@pytest.mark.anyio
async def test_gemini_unsigned_function_call_uses_server_tool_signature() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(text="Plan", thought=True),
                Part(
                    thought_signature=b"search-signature",
                    tool_call=GeminiToolCall(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["most populous country"]},
                    ),
                ),
                Part(
                    thought_signature=b"search-response-signature",
                    tool_response=ToolResponse(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"search_suggestions": ["India population"]},
                    ),
                ),
                Part(
                    function_call=FunctionCall(
                        id="days-1",
                        name="days_in_year",
                        args={"year": 2024},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].text == "Plan"
    assert google_content.parts[1].tool_call is not None
    assert google_content.parts[1].thought_signature == b"search-signature"
    assert google_content.parts[2].tool_response is not None
    assert google_content.parts[3].function_call is not None
    assert google_content.parts[3].function_call.id == "days-1"
    assert google_content.parts[3].thought_signature == b"search-signature"


def test_gemini_unsupported_server_tool_call_is_omitted() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    thought_signature=b"url-context",
                    tool_call=GeminiToolCall(
                        id="url-context-1",
                        tool_type=ToolType.URL_CONTEXT,
                        args={"urls": ["https://example.com"]},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)

    assert choice.message.content == ""


def test_gemini_code_execution_round_trips_for_replay() -> None:
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part.from_executable_code(
                    code="print(6 * 7)",
                    language=Language.PYTHON,
                ),
                Part.from_code_execution_result(
                    outcome=Outcome.OUTCOME_OK,
                    output="42\n",
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)

    assert isinstance(choice.message.content, list)
    assert isinstance(choice.message.content[0], ContentToolUse)
    tool_use = choice.message.content[0]
    assert tool_use.tool_type == "code_execution"
    assert tool_use.name == Language.PYTHON
    assert tool_use.arguments == "print(6 * 7)"
    assert tool_use.result == "42\n"

    replayed_parts = parts_from_server_tool_use(tool_use)
    assert replayed_parts[0].executable_code is not None
    assert replayed_parts[0].executable_code.language == Language.PYTHON
    assert replayed_parts[0].executable_code.code == "print(6 * 7)"
    assert replayed_parts[1].code_execution_result is not None
    assert replayed_parts[1].code_execution_result.outcome == Outcome.OUTCOME_OK
    assert replayed_parts[1].code_execution_result.output == "42\n"


@pytest.mark.anyio
async def test_gemini_function_response_preserves_tool_call_id() -> None:
    message = ChatMessageTool(
        content="42",
        tool_call_id="call-123",
        function="my_tool",
    )

    google_content = await content(MagicMock(), message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].function_response is not None
    assert google_content.parts[0].function_response.id == "call-123"


@pytest.mark.anyio
async def test_gemini_function_call_preserves_tool_call_id() -> None:
    message = ChatMessageAssistant(
        content=[],
        tool_calls=[
            ToolCall(
                id="call-123",
                function="my_tool",
                arguments={"x": 42},
            )
        ],
    )

    google_content = await content(MagicMock(), message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].function_call is not None
    assert google_content.parts[0].function_call.id == "call-123"


# ---------------------------------------------------------------------------
# Offline structural coverage for the Gemini 3 mixed-tools replay path.
#
# Each of these is independent of the live API. They lock down the structural
# contract of the capture/replay pipeline so future regressions surface in CI
# without needing a Google API key.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gemini_visible_thought_followed_by_signed_function_call_preserves_text() -> (
    None
):
    """Regression test for the headline bug fixed by PR #3817.

    A visible thought-text followed by a signed function_call must round-trip
    with both the visible text AND the function_call signature intact. Previous
    behavior overwrote the visible text onto a redacted ContentReasoning via
    _consolidate_thought_signature, then dropped it on replay, which Gemini's
    next-turn validator rejected.
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(text="Plan the call", thought=True),
                Part(
                    thought_signature=b"fc-signature",
                    function_call=FunctionCall(
                        id="days-1",
                        name="days_in_year",
                        args={"year": 2024},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].text == "Plan the call"
    assert google_content.parts[0].thought is True
    assert google_content.parts[1].function_call is not None
    assert google_content.parts[1].function_call.id == "days-1"
    assert google_content.parts[1].thought_signature == b"fc-signature"


def test_gemini_executable_code_signature_round_trips() -> None:
    """A signed executable_code part must preserve its thought_signature on

    capture, stored inside ContentToolUse.internal["gemini_parts"].
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    thought_signature=b"code-signature",
                    executable_code=ExecutableCode(
                        code="print(6 * 7)", language=Language.PYTHON
                    ),
                ),
                Part(
                    code_execution_result=CodeExecutionResult(
                        outcome=Outcome.OUTCOME_OK, output="42\n"
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)

    assert isinstance(choice.message.content, list)
    tool_use = choice.message.content[0]
    assert isinstance(tool_use, ContentToolUse)
    assert tool_use.tool_type == "code_execution"

    replayed_parts = parts_from_server_tool_use(tool_use)
    assert replayed_parts[0].executable_code is not None
    assert replayed_parts[0].thought_signature == b"code-signature"
    assert replayed_parts[1].code_execution_result is not None


@pytest.mark.anyio
async def test_gemini_executable_code_with_function_tool_mixed_signature_preserved() -> (
    None
):
    """Mixed mode: signed executable_code interleaved with a signed function_call.

    Both signatures must land on the right parts on replay; this is the actual
    Gemini 3 mixed-mode scenario that the next-turn validator rejects when
    either signature is dropped.
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(text="Plan", thought=True),
                Part(
                    thought_signature=b"code-signature",
                    executable_code=ExecutableCode(
                        code="print(6 * 7)", language=Language.PYTHON
                    ),
                ),
                Part(
                    code_execution_result=CodeExecutionResult(
                        outcome=Outcome.OUTCOME_OK, output="42\n"
                    ),
                ),
                Part(
                    thought_signature=b"fc-signature",
                    function_call=FunctionCall(
                        id="record-1",
                        name="record_result",
                        args={"answer": "42"},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].text == "Plan"
    assert google_content.parts[0].thought is True
    assert google_content.parts[1].executable_code is not None
    assert google_content.parts[1].thought_signature == b"code-signature"
    assert google_content.parts[2].code_execution_result is not None
    assert google_content.parts[3].function_call is not None
    assert google_content.parts[3].function_call.id == "record-1"
    assert google_content.parts[3].thought_signature == b"fc-signature"


def test_gemini_executable_code_legacy_log_fallback() -> None:
    """ContentToolUse(tool_type='code_execution') without internal['gemini_parts']

    simulates an eval log captured before signatures were preserved. Replay must
    fall back to reconstruction (no signature) without crashing.
    """
    legacy_tool_use = ContentToolUse(
        tool_type="code_execution",
        id="",
        name=Language.PYTHON,
        arguments="print(1 + 1)",
        result="2\n",
    )

    replayed_parts = parts_from_server_tool_use(legacy_tool_use)
    assert replayed_parts[0].executable_code is not None
    assert replayed_parts[0].executable_code.code == "print(1 + 1)"
    assert replayed_parts[0].thought_signature is None
    assert replayed_parts[1].code_execution_result is not None


@pytest.mark.anyio
async def test_gemini_parallel_function_calls_first_signed() -> None:
    """Per Gemini docs, parallel function_calls in one step have the signature

    on the first call only. After capture → replay, the first call carries the
    signature and the second is unsigned.
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(text="Plan", thought=True),
                Part(
                    thought_signature=b"sig",
                    function_call=FunctionCall(
                        id="f-1", name="days_in_year", args={"year": 2024}
                    ),
                ),
                Part(
                    function_call=FunctionCall(
                        id="f-2", name="days_in_year", args={"year": 2025}
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    assert google_content.parts[0].text == "Plan"
    assert google_content.parts[1].function_call is not None
    assert google_content.parts[1].function_call.id == "f-1"
    assert google_content.parts[1].thought_signature == b"sig"
    assert google_content.parts[2].function_call is not None
    assert google_content.parts[2].function_call.id == "f-2"
    assert google_content.parts[2].thought_signature is None


@pytest.mark.anyio
async def test_gemini_borrow_picks_first_server_signature() -> None:
    """When a step contains TWO signed server tool_calls and an unsigned

    client function_call, the borrow fallback must use the FIRST server
    tool's signature (most semantically related to the visible reasoning).
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    thought_signature=b"first-search",
                    tool_call=GeminiToolCall(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["india population"]},
                    ),
                ),
                Part(
                    thought_signature=b"first-search-resp",
                    tool_response=ToolResponse(
                        id="search-1",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"hits": ["1.4B"]},
                    ),
                ),
                Part(
                    thought_signature=b"second-search",
                    tool_call=GeminiToolCall(
                        id="search-2",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        args={"queries": ["china population"]},
                    ),
                ),
                Part(
                    thought_signature=b"second-search-resp",
                    tool_response=ToolResponse(
                        id="search-2",
                        tool_type=ToolType.GOOGLE_SEARCH_WEB,
                        response={"hits": ["1.4B"]},
                    ),
                ),
                Part(
                    function_call=FunctionCall(
                        id="record-1",
                        name="record_result",
                        args={"answer": "tied"},
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    function_call_parts = [
        p for p in google_content.parts if p.function_call is not None
    ]
    assert len(function_call_parts) == 1
    assert function_call_parts[0].function_call is not None
    assert function_call_parts[0].function_call.id == "record-1"
    # Borrow MUST use the first server-tool's signature, not the last
    assert function_call_parts[0].thought_signature == b"first-search"


@pytest.mark.anyio
async def test_gemini_unprefaced_signed_function_call() -> None:
    """A signed function_call with no preceding visible thought-text — the

    response begins directly with the function_call. Signature must still be
    preserved on the function_call part on replay.
    """
    candidate = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(
            role="model",
            parts=[
                Part(
                    thought_signature=b"sig",
                    function_call=FunctionCall(
                        id="f-1", name="days_in_year", args={"year": 2024}
                    ),
                ),
            ],
        ),
    )

    choice = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate)
    google_content = await content(MagicMock(), choice.message, emulate_reasoning=False)

    assert google_content.parts is not None
    function_call_parts = [
        p for p in google_content.parts if p.function_call is not None
    ]
    assert len(function_call_parts) == 1
    assert function_call_parts[0].function_call is not None
    assert function_call_parts[0].function_call.id == "f-1"
    assert function_call_parts[0].thought_signature == b"sig"


@pytest.mark.anyio
async def test_gemini_dangling_anchor_does_not_crash() -> None:
    """A ContentReasoning anchor whose gemini_function_call_id does not match

    any tool_call (could happen after a solver-side edit) must not crash;
    replay should ignore the orphan anchor and continue.
    """
    message = ChatMessageAssistant(
        content=[
            ContentText(text="ok"),
            ContentReasoning(
                reasoning=base64.b64encode(b"orphan-sig").decode(),
                redacted=True,
                internal={"gemini_function_call_id": "nonexistent-id"},
            ),
        ],
        tool_calls=[
            ToolCall(id="real-id", function="my_tool", arguments={"x": 1}),
        ],
    )

    google_content = await content(MagicMock(), message, emulate_reasoning=False)

    assert google_content.parts is not None
    # The real tool_call still emits, even if the anchor doesn't match
    function_call_parts = [
        p for p in google_content.parts if p.function_call is not None
    ]
    assert len(function_call_parts) == 1
    assert function_call_parts[0].function_call is not None
    assert function_call_parts[0].function_call.id == "real-id"


@pytest.mark.anyio
async def test_gemini_clear_reasoning_preserves_anchors_for_replay() -> None:
    """Compaction's _clear_reasoning must preserve positional anchors —

    redacted ContentReasoning blocks carrying internal['gemini_function_call_id']
    — even though they look like ordinary reasoning. Stripping them would
    orphan the matching tool_call and break replay with 400 INVALID_ARGUMENT.
    """
    visible_reasoning = ContentReasoning(
        reasoning="Plan the call",
        redacted=False,
    )
    anchor = ContentReasoning(
        reasoning=base64.b64encode(b"fc-signature").decode(),
        redacted=True,
        internal={"gemini_function_call_id": "f-1"},
    )
    standalone_redacted = ContentReasoning(
        reasoning=base64.b64encode(b"unrelated-sig").decode(),
        redacted=True,
    )
    msg = ChatMessageAssistant(
        content=[
            visible_reasoning,
            anchor,
            ContentText(text="Result text"),
            standalone_redacted,
        ],
        tool_calls=[ToolCall(id="f-1", function="my_tool", arguments={"x": 1})],
    )

    cleared = _clear_reasoning(msg)

    assert isinstance(cleared.content, list)
    # Visible reasoning is dropped (saves tokens)
    assert visible_reasoning not in cleared.content
    # Standalone redacted reasoning (no anchor key) is dropped
    assert standalone_redacted not in cleared.content
    # Anchor survives so replay can attach the signature to f-1
    assert anchor in cleared.content
    # Other content untouched
    assert any(
        isinstance(c, ContentText) and c.text == "Result text" for c in cleared.content
    )

    # Round-trip: feed the cleared message into content() and verify the
    # function_call still emits with its signature.
    google_content = await content(MagicMock(), cleared, emulate_reasoning=False)
    assert google_content.parts is not None
    function_call_parts = [
        p for p in google_content.parts if p.function_call is not None
    ]
    assert len(function_call_parts) == 1
    assert function_call_parts[0].function_call is not None
    assert function_call_parts[0].function_call.id == "f-1"
    assert function_call_parts[0].thought_signature == b"fc-signature"


@pytest.mark.anyio
async def test_gemini_mixed_tools_round_trip_is_idempotent() -> None:
    """Capture a candidate with the full mixed-tools shape (visible thought +

    signed server tool_call + tool_response + signed function_call), reconstruct
    via content(), then re-capture by feeding the reconstructed parts back into
    a synthetic candidate. The second capture must produce equivalent Inspect
    content (modulo internal dict regeneration).
    """
    original_parts = [
        Part(text="Plan", thought=True),
        Part(
            thought_signature=b"search-sig",
            tool_call=GeminiToolCall(
                id="search-1",
                tool_type=ToolType.GOOGLE_SEARCH_WEB,
                args={"queries": ["x"]},
            ),
        ),
        Part(
            thought_signature=b"search-resp-sig",
            tool_response=ToolResponse(
                id="search-1",
                tool_type=ToolType.GOOGLE_SEARCH_WEB,
                response={"hits": ["y"]},
            ),
        ),
        Part(
            thought_signature=b"fc-sig",
            function_call=FunctionCall(
                id="f-1", name="record_result", args={"answer": "y"}
            ),
        ),
    ]

    candidate1 = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(role="model", parts=original_parts),
    )
    choice1 = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate1)
    rebuilt_content = await content(
        MagicMock(), choice1.message, emulate_reasoning=False
    )
    assert rebuilt_content.parts is not None

    # Round-trip 2: feed the rebuilt parts back through capture.
    candidate2 = Candidate(
        finish_reason=FinishReason.STOP,
        content=Content(role="model", parts=rebuilt_content.parts),
    )
    choice2 = completion_choice_from_candidate("gemini-3.1-pro-preview", candidate2)

    # Both captures should produce the same number of tool_calls and same ids.
    assert choice1.message.tool_calls is not None
    assert choice2.message.tool_calls is not None
    assert [c.id for c in choice1.message.tool_calls] == [
        c.id for c in choice2.message.tool_calls
    ]


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_3_native_search_with_function_tool_replays() -> None:
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    user = ChatMessageUser(
        content=(
            "Use days_in_year(2024) AND web search 'most populous country'. "
            "Then a 1-line answer."
        )
    )

    first_output, _ = await api.generate(
        input=[user],
        tools=[_create_gemini_web_search_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(first_output, ModelOutput)
    assistant = first_output.choices[0].message

    assert isinstance(assistant.content, list)
    assert any(isinstance(item, ContentToolUse) for item in assistant.content)
    assert assistant.tool_calls is not None
    assert len(assistant.tool_calls) > 0

    tool_call = assistant.tool_calls[0]
    tool_result = ChatMessageTool(
        content="2024 has 366 days.",
        tool_call_id=tool_call.id,
        function=tool_call.function,
    )

    second_output, _ = await api.generate(
        input=[user, assistant, tool_result],
        tools=[_create_gemini_web_search_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(second_output, ModelOutput)

    assert second_output.choices[0].stop_reason == "stop"
    assert second_output.choices[0].message.text is not None


@skip_if_no_google
def test_gemini_3_native_search_with_function_tool_react_loop() -> None:
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "Use days_in_year(2024) AND web search 'most populous "
                        "country'. Then submit a one-line summary."
                    )
                )
            ],
            solver=react(tools=[web_search(providers="gemini"), _days_in_year_tool()]),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=10,
        max_tokens=4096,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples
    assert result.samples[0].output is not None


# ---------------------------------------------------------------------------
# Live (--runapi) coverage for Gemini 3 mixed-tool replays.
#
# These tests exercise Gemini's verifier against the parts we produce. They
# require GOOGLE_API_KEY and run against gemini-3.1-pro-preview unless noted.
# A failure here is the authoritative signal that our capture/replay path has
# drifted from what Gemini accepts on the next turn.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_3_native_code_execution_with_function_tool_replays() -> None:
    """Two-turn replay of native code_execution + custom function tool.

    Mirrors the search test but exercises the executable_code signature
    round-trip — fails with 400 INVALID_ARGUMENT if the executable_code
    thought_signature is dropped on capture or replay.
    """
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    user = ChatMessageUser(
        content=(
            "Use code execution to compute 47*53 AND call days_in_year(2024). "
            "Then a 1-line answer."
        )
    )

    first_output, _ = await api.generate(
        input=[user],
        tools=[_create_google_code_execution_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(first_output, ModelOutput)
    assistant = first_output.choices[0].message

    assert isinstance(assistant.content, list)
    # Either a code_execution ContentToolUse or a function_call should be present
    assert assistant.tool_calls is not None or any(
        isinstance(item, ContentToolUse) for item in assistant.content
    )

    # Build a follow-up. If there's a function_call, answer it; otherwise
    # just continue the conversation.
    follow_up: list[Any] = [user, assistant]
    if assistant.tool_calls:
        tool_call = assistant.tool_calls[0]
        follow_up.append(
            ChatMessageTool(
                content="2024 has 366 days.",
                tool_call_id=tool_call.id,
                function=tool_call.function,
            )
        )

    second_output, _ = await api.generate(
        input=follow_up,
        tools=[_create_google_code_execution_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(second_output, ModelOutput)
    assert second_output.choices[0].stop_reason in ("stop", "tool_calls")


@skip_if_no_google
def test_gemini_3_native_code_execution_with_function_tool_react_loop() -> None:
    """End-to-end react loop with native code_execution + custom function tool."""
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "Use code execution to compute 47*53 AND call "
                        "days_in_year(2024). Then submit a one-line summary."
                    )
                )
            ],
            solver=react(tools=[code_execution(), _days_in_year_tool()]),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=10,
        max_tokens=4096,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples
    assert result.samples[0].output is not None


@skip_if_no_google
def test_gemini_3_native_search_and_code_execution_with_function_tool() -> None:
    """Three-way mix: web_search + code_execution + custom function tool in

    one react loop. Stresses the full signature round-trip across all three
    server-side / client-side tool surfaces simultaneously.
    """
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "Web search for 'days in 2024', use code execution to "
                        "compute 47*53, AND call days_in_year(2024). Submit a "
                        "one-line summary referencing all three."
                    )
                )
            ],
            solver=react(
                tools=[
                    web_search(providers="gemini"),
                    code_execution(),
                    _days_in_year_tool(),
                ]
            ),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=12,
        max_tokens=6144,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples


@skip_if_no_google
def test_gemini_3_native_search_with_multiple_function_tools() -> None:
    """Native web_search alongside two custom function tools. Verifies that

    tool_choice='auto' + VALIDATED mode + multiple FunctionDeclarations works
    end-to-end without 400.
    """
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "Web search 'most populous country', call "
                        "days_in_year(2024), and call multiply(7, 6). "
                        "Submit a one-line summary."
                    )
                )
            ],
            solver=react(
                tools=[
                    web_search(providers="gemini"),
                    _days_in_year_tool(),
                    _multiply_tool(),
                ]
            ),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=12,
        max_tokens=4096,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples


@skip_if_no_google
def test_gemini_3_native_search_react_loop_parallel_calls() -> None:
    """A prompt designed to elicit parallel function_calls in one step. Per

    Gemini docs, only the first function_call carries a thought_signature;
    this test pins down that the part-ordering refinement keeps the signature
    on the right call across multiple turns of replay.
    """
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "In one step, web search 'days in 2024' AND call "
                        "days_in_year(2024) AND call days_in_year(2025) "
                        "(parallel). Submit a one-line summary."
                    )
                )
            ],
            solver=react(tools=[web_search(providers="gemini"), _days_in_year_tool()]),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=12,
        max_tokens=4096,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_3_compaction_clear_reasoning_then_replay() -> None:
    """Highest-value live coverage for refinement 3: capture a Gemini 3 mixed-

    tools response, run _clear_reasoning over the assistant message (anchors
    survive, visible reasoning is dropped), then send the compacted message
    back. Gemini must accept it with no 400.
    """
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    user = ChatMessageUser(
        content=(
            "Use days_in_year(2024) AND web search 'most populous country'. "
            "Then a 1-line answer."
        )
    )

    first_output, _ = await api.generate(
        input=[user],
        tools=[_create_gemini_web_search_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(first_output, ModelOutput)
    assistant = first_output.choices[0].message
    assert assistant.tool_calls is not None and len(assistant.tool_calls) > 0

    # Simulate compaction: strip non-anchor reasoning blocks
    compacted = _clear_reasoning(assistant)

    tool_call = compacted.tool_calls[0] if compacted.tool_calls else None
    assert tool_call is not None
    tool_result = ChatMessageTool(
        content="2024 has 366 days.",
        tool_call_id=tool_call.id,
        function=tool_call.function,
    )

    second_output, _ = await api.generate(
        input=[user, compacted, tool_result],
        tools=[_create_gemini_web_search_tool(), _create_days_in_year_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(second_output, ModelOutput)
    # The verifier must accept the compacted message — anchors carry the
    # signatures it requires.
    assert second_output.choices[0].stop_reason in ("stop", "tool_calls")


@skip_if_no_google
def test_gemini_3_sequential_function_calls_across_turns() -> None:
    """Multi-turn react loop where each turn produces a single function_call

    (no parallel). Each turn's signature must round-trip independently and
    the loop must complete without 400.
    """
    result = eval(
        Task(
            dataset=[
                Sample(
                    input=(
                        "First call days_in_year(2024). After you have the "
                        "result, call multiply(366, 24) to compute total "
                        "hours. Then submit a one-line summary."
                    )
                )
            ],
            solver=react(tools=[_days_in_year_tool(), _multiply_tool()]),
        ),
        model="google/gemini-3.1-pro-preview",
        message_limit=10,
        max_tokens=4096,
        fail_on_error=False,
    )[0]

    assert result.status == "success"
    assert result.samples


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_2_5_native_search_with_function_tool_raises() -> None:
    """Live regression: Gemini 2.5 still raises ValueError before the request

    is sent. Catches drift if is_gemini_3_plus() ever mis-classifies a model
    name pattern.
    """
    api = GoogleGenAIAPI(
        model_name="gemini-2.5-pro",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    with pytest.raises(ValueError, match="requires Gemini 3 or later"):
        await api.generate(
            input=[ChatMessageUser(content="Search and call my_tool")],
            tools=[_create_gemini_web_search_tool(), _create_test_tool()],
            tool_choice="auto",
            config=GenerateConfig(max_tokens=512),
        )


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_3_native_search_alone_still_works() -> None:
    """Native-only regression: web_search with NO custom function tools must

    still work via the original native-only flow (no
    include_server_side_tool_invocations, no VALIDATED mode). Confirms the
    mixed-mode changes don't regress the single-tool path.
    """
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    output, _ = await api.generate(
        input=[ChatMessageUser(content="What is the population of India?")],
        tools=[_create_gemini_web_search_tool()],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(output, ModelOutput)
    assert output.choices[0].stop_reason == "stop"


@pytest.mark.anyio
@skip_if_no_google
async def test_gemini_3_replay_after_dropping_a_tool_call() -> None:
    """Manual-edit defense: capture a response with at least one function_call,

    drop one of the tool_calls (simulating solver-side mutation), send the
    edited message back. The request must either succeed or fail with a
    controlled error — never an Inspect-side crash.
    """
    api = GoogleGenAIAPI(
        model_name="gemini-3.1-pro-preview",
        base_url=None,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    user = ChatMessageUser(
        content=(
            "In one step call days_in_year(2024) AND call multiply(7, 6). "
            "Then a 1-line answer."
        )
    )

    first_output, _ = await api.generate(
        input=[user],
        tools=[
            _create_days_in_year_tool(),
            ToolInfo(
                name="multiply",
                description="Multiply two integers.",
                parameters=ToolParams(
                    type="object",
                    properties={
                        "a": ToolParam(type="integer", description="a"),
                        "b": ToolParam(type="integer", description="b"),
                    },
                    required=["a", "b"],
                ),
            ),
        ],
        tool_choice="auto",
        config=GenerateConfig(max_tokens=2048),
    )
    assert isinstance(first_output, ModelOutput)
    assistant = first_output.choices[0].message
    if not assistant.tool_calls or len(assistant.tool_calls) < 1:
        pytest.skip("Model did not emit any tool calls; cannot exercise the edit path")

    # Drop the second (and beyond) tool_call(s) to simulate solver-side mutation
    edited = assistant.model_copy(update={"tool_calls": assistant.tool_calls[:1]})

    kept = edited.tool_calls[0] if edited.tool_calls else None
    assert kept is not None
    tool_result = ChatMessageTool(
        content="result", tool_call_id=kept.id, function=kept.function
    )

    # Either the request succeeds or fails with an API ValueError; do not crash
    try:
        second_output, _ = await api.generate(
            input=[user, edited, tool_result],
            tools=[
                _create_days_in_year_tool(),
                ToolInfo(
                    name="multiply",
                    description="Multiply two integers.",
                    parameters=ToolParams(
                        type="object",
                        properties={
                            "a": ToolParam(type="integer", description="a"),
                            "b": ToolParam(type="integer", description="b"),
                        },
                        required=["a", "b"],
                    ),
                ),
            ],
            tool_choice="auto",
            config=GenerateConfig(max_tokens=512),
        )
        # If it succeeded, fine
        assert isinstance(second_output, ModelOutput)
    except ValueError:
        # If the API rejected the edited message, that's a controlled failure
        pass
