from inspect_ai._util.content import (
    ContentAudio,
    ContentImage,
    ContentMarkdown,
    ContentVideo,
)
from inspect_ai.event import MediaEvent
from inspect_ai.log._transcript import Transcript


def test_media_event_image_serialization():
    """Test round-trip serialization of MediaEvent with image content."""
    original = MediaEvent(content=ContentImage(image="data:image/png;base64,abc"))
    serialized = original.model_dump_json()
    deserialized = MediaEvent.model_validate_json(serialized)
    assert original.content == deserialized.content
    assert deserialized.event == "media"


def test_media_event_video_serialization():
    """Test round-trip serialization of MediaEvent with video content."""
    original = MediaEvent(
        content=ContentVideo(video="test.mp4", format="mp4"), caption="Demo"
    )
    serialized = original.model_dump_json()
    deserialized = MediaEvent.model_validate_json(serialized)
    assert original.content == deserialized.content
    assert deserialized.caption == "Demo"


def test_media_event_audio_serialization():
    """Test round-trip serialization of MediaEvent with audio content."""
    original = MediaEvent(
        content=ContentAudio(audio="test.mp3", format="mp3"), source="recorder"
    )
    serialized = original.model_dump_json()
    deserialized = MediaEvent.model_validate_json(serialized)
    assert original.content == deserialized.content
    assert deserialized.source == "recorder"


def test_media_event_markdown_serialization():
    """Test round-trip serialization of MediaEvent with markdown content."""
    original = MediaEvent(
        content=ContentMarkdown(markdown="# Hello\n\nThis is **bold**."),
        caption="Documentation",
    )
    serialized = original.model_dump_json()
    deserialized = MediaEvent.model_validate_json(serialized)
    assert original.content == deserialized.content
    assert deserialized.caption == "Documentation"


def test_transcript_image():
    """Test transcript().image() convenience method."""
    t = Transcript()
    t.image("https://example.com/img.png", caption="Test", source="test")
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.event == "media"
    assert event.content.type == "image"
    assert event.content.image == "https://example.com/img.png"
    assert event.caption == "Test"
    assert event.source == "test"


def test_transcript_video():
    """Test transcript().video() convenience method."""
    t = Transcript()
    t.video("test.mp4", format="mp4", caption="Demo")
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.content.type == "video"
    assert event.content.video == "test.mp4"
    assert event.content.format == "mp4"
    assert event.caption == "Demo"


def test_transcript_audio():
    """Test transcript().audio() convenience method."""
    t = Transcript()
    t.audio("test.mp3", format="mp3", source="recorder")
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.content.type == "audio"
    assert event.content.audio == "test.mp3"
    assert event.content.format == "mp3"
    assert event.source == "recorder"


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


def test_transcript_markdown():
    """Test transcript().markdown() convenience method."""
    t = Transcript()
    t.markdown("# Title\n\nSome **bold** text.", caption="Notes", source="agent")
    assert len(t.events) == 1
    event = t.events[0]
    assert isinstance(event, MediaEvent)
    assert event.content.type == "markdown"
    assert event.content.markdown == "# Title\n\nSome **bold** text."
    assert event.caption == "Notes"
    assert event.source == "agent"
