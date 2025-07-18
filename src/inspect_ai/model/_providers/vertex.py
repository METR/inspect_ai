# type: ignore

import functools
import json
from copy import copy
from typing import Any, cast

import vertexai  # type: ignore
from google.api_core.exceptions import (
    Aborted,
    ClientError,
    DeadlineExceeded,
    ServiceUnavailable,
)
from google.api_core.retry import if_transient_error
from google.protobuf.json_format import MessageToDict
from pydantic import JsonValue
from typing_extensions import override
from vertexai.generative_models import (  # type: ignore
    Candidate,
    FinishReason,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part,
    Tool,
)
from vertexai.generative_models import Content as VertexContent

from inspect_ai._util.constants import BASE_64_DATA_REMOVED, NO_CONTENT
from inspect_ai._util.content import (
    Content,
    ContentAudio,
    ContentData,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai._util.http import is_retryable_http_status
from inspect_ai._util.images import file_as_data
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from .._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from .._generate_config import GenerateConfig
from .._model import ModelAPI
from .._model_call import ModelCall
from .._model_output import (
    ChatCompletionChoice,
    Logprob,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
    TopLogprob,
)

SAFETY_SETTINGS = "safety_settings"
VERTEX_INIT_ARGS = "vertex_init_args"

DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class VertexAPI(ModelAPI):
    """
    Provider for using the Google Vertex AI model endpoint.

    Note that this is an alternative to the provider implemented in `google.py`
    which deals with endpoints for Google AI Studio. If in doubt which one to
    use, you probably want `google.py`, you can see a comparison matrix here:
    https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/migrate-google-ai

    By default we assume the environment we run in is authenticated
    appropriately, but you can pass in `vertex_init_args` to configure this
    directly, this is passed directly to `vertex.init` which is documented here:
    https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai#vertexai_init

    Additional `model_args`:
        vertex_init_args (dict[str, Any]): Additional arguments to pass to `vertexai.init`
        safety_settings (dict[str, str]): Mapping for adjusting Gemini safety settings
    """

    def __init__(
        self,
        model_name: str,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, config=config)

        if VERTEX_INIT_ARGS in model_args:
            vertexai.init(**model_args[VERTEX_INIT_ARGS])
            del model_args[VERTEX_INIT_ARGS]
        else:
            vertexai.init()

        # pick out vertex safety settings and merge against default
        self.safety_settings = DEFAULT_SAFETY_SETTINGS.copy()
        if SAFETY_SETTINGS in model_args:
            self.safety_settings.update(
                parse_safety_settings(model_args.get(SAFETY_SETTINGS))
            )
            del model_args[SAFETY_SETTINGS]

        self.model = GenerativeModel(model_name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        parameters = GenerationConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_output_tokens=config.max_tokens,
            stop_sequences=config.stop_seqs,
            candidate_count=config.num_choices,
            seed=config.seed,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            response_logprobs=config.logprobs,
            logprobs=config.top_logprobs,
        )

        messages = await as_chat_messages(input)
        vertex_tools = chat_tools(tools) if len(tools) > 0 else None

        response = await self.model.generate_content_async(
            contents=messages,
            safety_settings=self.safety_settings,
            generation_config=parameters,
            tools=vertex_tools,
        )

        # capture output
        output = ModelOutput(
            model=self.model_name,
            choices=completion_choices_from_candidates(
                self.model_name, response.candidates
            ),
            usage=ModelUsage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count,
            ),
        )

        # build call
        call = model_call(
            contents=messages,
            safety_settings=self.safety_settings,
            generation_config=parameters,
            tools=vertex_tools,
            response=response,
        )

        # return
        return output, call

    @override
    def should_retry(self, ex: Exception) -> bool:
        # google API-specific errors
        if isinstance(ex, Aborted | DeadlineExceeded | ServiceUnavailable):
            return True
        # standard HTTP errors
        elif isinstance(ex, ClientError) and ex.code is not None:
            return is_retryable_http_status(ex.code)
        # additional errors flagged by google as transient
        elif isinstance(ex, Exception):
            return if_transient_error(ex)
        else:
            return False

    @override
    def connection_key(self) -> str:
        """Scope for enforcing max_connections (could also use endpoint)."""
        return self.model_name

    @override
    def collapse_user_messages(self) -> bool:
        return True

    @override
    def collapse_assistant_messages(self) -> bool:
        return True


def model_call(
    contents: list[Content],
    generation_config: GenerationConfig,
    safety_settings: dict[HarmCategory, HarmBlockThreshold],
    tools: list[Tool] | None,
    response: GenerationResponse,
) -> ModelCall:
    return ModelCall.create(
        request=dict(
            contents=[model_call_content(content) for content in contents],
            generation_config=generation_config.to_dict(),
            safety_settings=safety_settings,
            tools=[tool.to_dict() for tool in tools] if tools is not None else None,
        ),
        response=response.to_dict(),
        filter=model_call_filter,
    )


def model_call_filter(key: JsonValue | None, value: JsonValue) -> JsonValue:
    # remove images from raw api call
    if key == "inline_data" and isinstance(value, dict) and "data" in value:
        value = copy(value)
        value.update(data=BASE_64_DATA_REMOVED)
    return value


def model_call_content(content: VertexContent) -> dict[str, Any]:
    return cast(dict[str, Any], content.to_dict())


async def as_chat_messages(messages: list[ChatMessage]) -> list[VertexContent]:
    # google does not support system messages so filter them out to start with
    system_messages = [message for message in messages if message.role == "system"]
    supported_messages = [message for message in messages if message.role != "system"]

    # build google chat messages
    chat_messages: list[VertexContent] = [
        await content_dict(message) for message in supported_messages
    ]

    # we want the system messages to be prepended to the first user message
    # (if there is no first user message then prepend one)
    prepend_system_messages(chat_messages, system_messages)

    # combine consecutive tool messages
    chat_messages = functools.reduce(consective_tool_message_reducer, chat_messages, [])

    # return messages
    return chat_messages


def consective_tool_message_reducer(
    messages: list[VertexContent],
    message: VertexContent,
) -> list[VertexContent]:
    if (
        message.role == "function"
        and len(messages) > 0
        and messages[-1].role == "function"
    ):
        messages[-1] = VertexContent(
            role="function", parts=messages[-1].parts + message.parts
        )
    else:
        messages.append(message)
    return messages


async def content_dict(
    message: ChatMessageUser | ChatMessageAssistant | ChatMessageTool,
) -> VertexContent:
    if isinstance(message, ChatMessageUser):
        if isinstance(message.content, str):
            parts = [Part.from_text(message.content or NO_CONTENT)]
        else:
            parts = [await content_part(content) for content in message.content]

        return VertexContent(role="user", parts=parts)
    elif isinstance(message, ChatMessageAssistant):
        content_parts: list[Part] = []
        if message.tool_calls is not None:
            content_parts.extend(
                [
                    # For some reason there's no `Parts.from_function_call`
                    # function, but there's a generic `from_dict` instead
                    Part.from_dict(
                        {
                            "function_call": {
                                "name": tool_call.function,
                                "args": tool_call.arguments,
                            }
                        }
                    )
                    for tool_call in message.tool_calls
                ]
            )

        if isinstance(message.content, str):
            content_parts.append(Part.from_text(message.content or NO_CONTENT))
        else:
            content_parts.extend(
                [await content_part(content) for content in message.content]
            )

        return VertexContent(role="model", parts=content_parts)

    elif isinstance(message, ChatMessageTool):
        return VertexContent(
            role="function",
            parts=[
                Part.from_function_response(
                    name=message.tool_call_id,
                    response={
                        "content": (
                            message.error.message
                            if message.error is not None
                            else message.text
                        )
                    },
                )
            ],
        )


async def content_part(content: Content | str) -> Part:
    if isinstance(content, str):
        return Part.from_text(content or NO_CONTENT)
    elif isinstance(content, ContentText):
        return Part.from_text(content.text or NO_CONTENT)
    elif isinstance(content, ContentImage):
        image_bytes, mime_type = await file_as_data(content.image)
        return Part.from_image(image=Image.from_bytes(data=image_bytes))
    elif isinstance(content, ContentReasoning):
        return Part.from_text(content.reasoning or NO_CONTENT)
    else:
        if isinstance(content, ContentAudio):
            file = content.audio
        elif isinstance(content, ContentData):
            file = ""
            assert False, "Vertex provider should never encounter ContentData"
        else:
            # it's ContentVideo
            file = content.video
        file_bytes, mime_type = await file_as_data(file)
        return Part.from_data(file_bytes, mime_type)


def prepend_system_messages(
    messages: list[VertexContent], system_messages: list[ChatMessageSystem]
) -> None:
    # create system_parts
    system_parts = [Part.from_text(message.content) for message in system_messages]

    # we want the system messages to be prepended to the first user message
    # (if there is no first user message then prepend one)
    if messages[0].role == "user":
        parts = messages[0].parts
        messages[0] = VertexContent(role="user", parts=system_parts + parts)
    else:
        messages.insert(0, VertexContent(role="user", parts=system_parts))


def chat_tools(tools: list[ToolInfo]) -> list[Tool]:
    declarations = [
        FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters.model_dump(
                exclude_none=True, exclude={"additionalProperties"}
            ),
        )
        for tool in tools
    ]
    return [Tool(function_declarations=declarations)]


def completion_choice_from_candidate(
    model: str, candidate: Candidate
) -> ChatCompletionChoice:
    # check for completion text
    content = " ".join(
        [
            part.text
            for part in candidate.content.parts
            if part.to_dict().get("text") is not None
        ]
    )

    # now tool calls
    tool_calls: list[ToolCall] = []
    for part in candidate.content.parts:
        if part.function_call:
            function_call = MessageToDict(getattr(part.function_call, "_pb"))
            tool_calls.append(
                ToolCall(
                    id=function_call["name"],
                    function=function_call["name"],
                    arguments=function_call["args"],
                )
            )

    # stop reason
    stop_reason = candidate_stop_reason(candidate.finish_reason)

    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(
            content=content,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
            model=model,
            source="generate",
        ),
        stop_reason=stop_reason,
    )

    if candidate.logprobs_result:
        logprobs: list[Logprob] = []
        for chosen, top in zip(
            candidate.logprobs_result.chosen_candidates,
            candidate.logprobs_result.top_candidates,
        ):
            logprobs.append(
                Logprob(
                    token=chosen.token,
                    logprob=chosen.log_probability,
                    top_logprobs=[
                        TopLogprob(token=c.token, logprob=c.log_probability)
                        for c in top.candidates
                    ],
                )
            )
        choice.logprobs = Logprobs(content=logprobs)

    return choice


def completion_choices_from_candidates(
    model: str,
    candidates: list[Candidate],
) -> list[ChatCompletionChoice]:
    candidates = copy(candidates)
    candidates.sort(key=lambda c: c.index)
    return [
        completion_choice_from_candidate(model, candidate) for candidate in candidates
    ]


def candidate_stop_reason(finish_reason: FinishReason) -> StopReason:
    match finish_reason:
        case FinishReason.STOP:
            return "stop"
        case FinishReason.MAX_TOKENS:
            return "max_tokens"
        case FinishReason.SAFETY | FinishReason.RECITATION:
            return "content_filter"
        case _:
            return "unknown"


def parse_safety_settings(
    safety_settings: Any,
) -> dict[HarmCategory, HarmBlockThreshold]:
    # ensure we have a dict
    if isinstance(safety_settings, str):
        safety_settings = json.loads(safety_settings)
    if not isinstance(safety_settings, dict):
        raise ValueError(f"{SAFETY_SETTINGS} must be dictionary.")

    parsed_settings: dict[HarmCategory, HarmBlockThreshold] = {}
    for key, value in safety_settings.items():
        if isinstance(key, str):
            key = str_to_harm_category(key)
        if not isinstance(key, HarmCategory):
            raise ValueError(f"Unexpected type for harm category: {key}")
        if isinstance(value, str):
            value = str_to_harm_block_threshold(value)
        if not isinstance(value, HarmBlockThreshold):
            raise ValueError(f"Unexpected type for harm block threshold: {value}")

        parsed_settings[key] = value

    return parsed_settings


def str_to_harm_category(category: str) -> HarmCategory:
    category = category.upper()
    if "HARASSMENT" in category:
        return HarmCategory.HARM_CATEGORY_HARASSMENT
    elif "HATE_SPEECH" in category:
        return HarmCategory.HARM_CATEGORY_HATE_SPEECH
    elif "SEXUALLY_EXPLICIT" in category:
        return HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
    elif "DANGEROUS_CONTENT" in category:
        return HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
    elif "UNSPECIFIED" in category:
        return HarmCategory.HARM_CATEGORY_UNSPECIFIED
    else:
        raise ValueError(f"Unknown HarmCategory: {category}")


def str_to_harm_block_threshold(threshold: str) -> HarmBlockThreshold:
    threshold = threshold.upper()
    if "LOW" in threshold:
        return HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    elif "MEDIUM" in threshold:
        return HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    elif "HIGH" in threshold:
        return HarmBlockThreshold.BLOCK_ONLY_HIGH
    elif "NONE" in threshold:
        return HarmBlockThreshold.BLOCK_NONE
    else:
        raise ValueError(f"Unknown HarmBlockThreshold: {threshold}")
