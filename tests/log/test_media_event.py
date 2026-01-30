import pytest

from inspect_ai._util.content import (
    ContentVideo,
)
from inspect_ai.event import MediaEvent
from inspect_ai.log._transcript import Transcript


@pytest.mark.parametrize(
    "method,args,kwargs,expected_type,field_name,field_value",
    [
        (
            "image",
            ("https://example.com/img.png",),
            {"caption": "Test", "source": "test"},
            "image",
            "image",
            "https://example.com/img.png",
        ),
        (
            "video",
            ("test.mp4", "mp4"),
            {"caption": "Demo"},
            "video",
            "video",
            "test.mp4",
        ),
        (
            "audio",
            ("test.mp3", "mp3"),
            {"source": "recorder"},
            "audio",
            "audio",
            "test.mp3",
        ),
        (
            "markdown",
            ("# Title\n\nSome **bold** text.",),
            {"caption": "Notes", "source": "agent"},
            "markdown",
            "markdown",
            "# Title\n\nSome **bold** text.",
        ),
    ],
)
def test_transcript_media_methods(
    method, args, kwargs, expected_type, field_name, field_value
):
    """Test transcript convenience methods create correct MediaEvents."""
    t = Transcript()
    getattr(t, method)(*args, **kwargs)
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.content.type == expected_type
    assert getattr(event.content, field_name) == field_value


def test_transcript_media_generic():
    """Test transcript().media() generic method."""
    t = Transcript()
    content = ContentVideo(video="x.mp4", format="mp4")
    t.media(content, caption="Generic", source="test")
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.content == content
    assert event.caption == "Generic"
    assert event.source == "test"
