from typing import Literal, Union

from pydantic import Field

from inspect_ai._util.content import (
    ContentAudio,
    ContentImage,
    ContentMarkdown,
    ContentVideo,
)
from inspect_ai.event._base import BaseEvent

MediaContent = Union[ContentImage, ContentAudio, ContentVideo, ContentMarkdown]
"""Media content types (image, audio, video, or markdown)."""


class MediaEvent(BaseEvent):
    """Event with media content (image, audio, or video)."""

    event: Literal["media"] = Field(default="media")
    """Event type."""

    content: MediaContent
    """Media content."""

    caption: str | None = Field(default=None)
    """Optional caption for the media."""

    source: str | None = Field(default=None)
    """Optional source for media event."""
