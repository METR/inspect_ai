from typing import Any, Callable, NamedTuple, Sequence, Type

from pydantic_core import to_json
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widget import Widget
from textual.widgets import Static

from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai._util.rich import lines_display
from inspect_ai._util.transcript import (
    set_transcript_markdown_options,
    transcript_function,
    transcript_markdown,
    transcript_reasoning,
    transcript_separator,
)
from inspect_ai.log._samples import ActiveSample
from inspect_ai.log._transcript import (
    ApprovalEvent,
    ErrorEvent,
    Event,
    InfoEvent,
    InputEvent,
    LoggerEvent,
    ModelEvent,
    SampleInitEvent,
    SampleLimitEvent,
    ScoreEvent,
    SpanBeginEvent,
    SubtaskEvent,
    ToolEvent,
)
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageUser,
)
from inspect_ai.model._render import messages_preceding_assistant
from inspect_ai.tool._tool import ToolResult
from inspect_ai.tool._tool_transcript import transcript_tool_call


class TranscriptView(ScrollableContainer):
    DEFAULT_CSS = """
    TranscriptView {
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._sample_id: str | None = None
        self._sample_events: int | None = None

        self._active = False
        self._pending_sample: ActiveSample | None = None

    async def notify_active(self, active: bool) -> None:
        self._active = active
        if self._active and self._pending_sample:
            await self.sync_sample(self._pending_sample)
            self._pending_sample = None

    async def sync_sample(self, sample: ActiveSample | None) -> None:
        # if sample is none then reset
        if sample is None:
            self._sample = None
            self._sample_events = None
            await self.remove_children()

        # process sample if we are active
        elif self._active:
            # if we have either a new sample or a new event count then proceed
            if (
                sample.id != self._sample_id
                or len(sample.transcript.events) != self._sample_events
            ):
                # update (scrolling to end if we are already close to it)
                new_sample = sample.id != self._sample_id
                scroll_to_end = (
                    new_sample or abs(self.scroll_y - self.max_scroll_y) <= 20
                )

                async with self.batch():
                    await self.remove_children()
                    await self.mount_all(
                        self._widgets_for_events(sample.transcript.events)
                    )
                if scroll_to_end:
                    self.scroll_end(animate=not new_sample)

                # set members
                self._sample_id = sample.id
                self._sample_events = len(sample.transcript.events)

        # if we aren't active then save as a pending sample
        else:
            self._pending_sample = sample

    def _widgets_for_events(
        self, events: Sequence[Event], limit: int = 10
    ) -> list[Widget]:
        widgets: list[Widget] = []

        # filter the events to the <limit> most recent
        filtered_events = events
        if len(events) > limit:
            filtered_events = filtered_events[-limit:]

        # find the sample init event
        sample_init: SampleInitEvent | None = None
        for event in events:
            if isinstance(event, SampleInitEvent):
                sample_init = event
                break

        # add the sample init event if it isn't already in the event list
        if sample_init and sample_init not in filtered_events:
            filtered_events = [sample_init] + list(filtered_events)

        # compute how many events we filtered out
        filtered_count = len(events) - len(filtered_events)
        showed_filtered_count = False

        for event in filtered_events:
            display = render_event(event)
            if display:
                for d in display:
                    if d.content:
                        widgets.append(
                            Static(
                                transcript_separator(
                                    d.title, self.app.current_theme.primary
                                )
                            )
                        )
                        if isinstance(d.content, Markdown):
                            set_transcript_markdown_options(d.content)
                        widgets.append(Static(d.content, markup=False))
                        widgets.append(Static(Text(" ")))

                        if not showed_filtered_count and filtered_count > 0:
                            showed_filtered_count = True

                            widgets.append(
                                Static(
                                    transcript_separator(
                                        f"{filtered_count} events..."
                                        if filtered_count > 1
                                        else "1 event...",
                                        self.app.current_theme.primary,
                                    )
                                )
                            )
                            widgets.append(Static(Text(" ")))

        return widgets


class EventDisplay(NamedTuple):
    """Display for an event group."""

    title: str
    """Text for title bar"""

    content: RenderableType | None = None
    """Optional custom content to display."""


def render_event(event: Event) -> list[EventDisplay] | None:
    # see if we have a renderer
    for event_type, renderer in _renderers:
        if isinstance(event, event_type):
            display = renderer(event)
            if display is not None:
                return display if isinstance(display, list) else [display]

    # no renderer
    return None


def render_sample_init_event(event: SampleInitEvent) -> EventDisplay:
    # alias sample
    sample = event.sample

    # input
    messages: list[ChatMessage] = (
        [ChatMessageUser(content=sample.input)]
        if isinstance(sample.input, str)
        else sample.input
    )
    content: list[RenderableType] = []
    for message in messages:
        content.extend(render_message(message))

    # target
    if sample.target:
        content.append(Text())
        content.append(Text("Target", style="bold"))
        content.append(Text())
        content.append(str(sample.target).strip())

    return EventDisplay("sample init", Group(*content))


def render_sample_limit_event(event: SampleLimitEvent) -> EventDisplay:
    return EventDisplay(f"limit: {event.type}", Text(event.message))


def render_model_event(event: ModelEvent) -> EventDisplay:
    # content
    content: list[RenderableType] = []

    def append_message(message: ChatMessage) -> None:
        content.extend(render_message(message))

    # render preceding messages
    preceding = messages_preceding_assistant(event.input)
    for message in preceding:
        append_message(message)
        content.append(Text())

    # display assistant message (note that we don't render tool calls
    # because they will be handled as part of render_tool)
    if event.output.message and event.output.message.text:
        append_message(event.output.message)

    return EventDisplay(f"model: {event.model}", Group(*content))


def render_sub_events(events: list[Event]) -> list[RenderableType]:
    content: list[RenderableType] = []
    for e in events:
        event_displays = render_event(e) or []
        for d in event_displays:
            if d.content:
                content.append(Text("  "))
                content.append(transcript_separator(d.title, "black", "··"))
                if isinstance(d.content, Markdown):
                    set_transcript_markdown_options(d.content)
                content.append(d.content)

    return content


def render_tool_event(event: ToolEvent) -> list[EventDisplay]:
    # render the call
    content = transcript_tool_call(event)

    # render the output
    if isinstance(event.result, list):
        result: ToolResult = "\n".join(
            [
                content.text
                for content in event.result
                if isinstance(content, ContentText)
            ]
        )
    else:
        result = event.result

    if result:
        content.append(Text())
        result = str(result).strip()
        content.extend(lines_display(result, 50))

    return [EventDisplay("tool call", Group(*content))]


def render_score_event(event: ScoreEvent) -> EventDisplay:
    table = Table(box=None, show_header=False)
    table.add_column("", min_width=10, justify="left")
    table.add_column("", justify="left")
    table.add_row("Target", str(event.target).strip())
    if event.score.answer:
        table.add_row("Answer", transcript_markdown(event.score.answer, escape=True))
    table.add_row("Score", str(event.score.value).strip())
    if event.score.explanation:
        table.add_row(
            "Explanation", transcript_markdown(event.score.explanation, escape=True)
        )

    return EventDisplay("score", table)


def render_subtask_event(event: SubtaskEvent) -> list[EventDisplay]:
    # render header
    content: list[RenderableType] = [transcript_function(event.name, event.input)]

    if event.result:
        content.append(Text())
        if isinstance(event.result, str | int | float | bool | None):
            content.append(Text(str(event.result)))
        else:
            content.append(render_as_json(event.result))

    return [EventDisplay(f"subtask: {event.name}", Group(*content))]


def render_input_event(event: InputEvent) -> EventDisplay:
    return EventDisplay("input", Text.from_ansi(event.input_ansi.strip()))


def render_approval_event(event: ApprovalEvent) -> EventDisplay:
    content: list[RenderableType] = [
        f"[bold]{event.approver}[/bold]: {event.decision} ({event.explanation})"
    ]

    return EventDisplay("approval", Group(*content))


def render_info_event(event: InfoEvent) -> EventDisplay:
    if isinstance(event.data, str):
        content: RenderableType = transcript_markdown(event.data)
    else:
        content = render_as_json(event.data)
    return EventDisplay("info", content)


def render_logger_event(event: LoggerEvent) -> EventDisplay:
    content = event.message.level.upper()
    if event.message.name:
        content = f"{content} (${event.message.name})"
    content = f"{content}: {event.message.message}"
    return EventDisplay("logger", content)


def render_error_event(event: ErrorEvent) -> EventDisplay:
    return EventDisplay("error", event.error.traceback.strip())


def render_as_json(json: Any) -> RenderableType:
    return transcript_markdown(
        "```json\n"
        + to_json(json, indent=2, fallback=lambda _: None).decode()
        + "\n```\n"
    )


def render_message(message: ChatMessage) -> list[RenderableType]:
    content: list[RenderableType] = [
        Text(message.role.capitalize(), style="bold"),
        Text(),
    ]

    # deal with plain text or with content blocks
    if isinstance(message.content, str):
        content.extend([transcript_markdown(message.text.strip(), escape=True)])
    else:
        for c in message.content:
            if isinstance(c, ContentReasoning):
                content.extend(transcript_reasoning(c))
            elif isinstance(c, ContentText):
                content.extend([transcript_markdown(c.text.strip(), escape=True)])

    return content


def span_title(event: SpanBeginEvent) -> str:
    return f"{event.type or 'span'}: {event.name}"


EventRenderer = Callable[[Any], EventDisplay | list[EventDisplay] | None]

_renderers: list[tuple[Type[Event], EventRenderer]] = [
    (SampleInitEvent, render_sample_init_event),
    (SampleLimitEvent, render_sample_limit_event),
    (ModelEvent, render_model_event),
    (ToolEvent, render_tool_event),
    (SubtaskEvent, render_subtask_event),
    (ScoreEvent, render_score_event),
    (InputEvent, render_input_event),
    (ApprovalEvent, render_approval_event),
    (InfoEvent, render_info_event),
    (LoggerEvent, render_logger_event),
    (ErrorEvent, render_error_event),
]
