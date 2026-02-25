import pathlib
from typing import Literal

import pytest

from inspect_ai._util.constants import LOG_SCHEMA_VERSION
from inspect_ai.event import ModelEvent, SampleInitEvent
from inspect_ai.log._convert import convert_eval_logs
from inspect_ai.log._file import read_eval_log
from inspect_ai.log._log import EvalLog

_TESTS_DIR = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "stream", [True, False, 3], ids=["stream", "no-stream", "stream-3"]
)
@pytest.mark.parametrize("to", ["eval", "json"])
@pytest.mark.parametrize(
    "resolve_attachments",
    ["full", "core", False],
    ids=["resolve-attachments", "resolve-core-attachments", "no-resolve-attachments"],
)
def test_convert_eval_logs(
    tmp_path: pathlib.Path,
    stream: bool | int,
    to: Literal["eval", "json"],
    resolve_attachments: bool | Literal["full", "core"],
):
    input_file = (
        _TESTS_DIR
        / "test_list_logs/2024-11-05T13-32-37-05-00_input-task_hxs4q9azL3ySGkjJirypKZ.eval"
    )

    convert_eval_logs(
        str(input_file),
        to,
        str(tmp_path),
        resolve_attachments=resolve_attachments,
        stream=stream,
    )

    output_file = (tmp_path / input_file.name).with_suffix(f".{to}")
    assert output_file.exists()

    # Read with resolve_attachments="full" to verify content is accessible after
    # round-trip (convert always condenses on write, so raw content has attachment refs)
    log = read_eval_log(str(output_file), resolve_attachments="full")
    assert isinstance(
        log,
        EvalLog,
    )
    assert log.samples
    assert log.samples[0].events
    sample_init_event = log.samples[0].events[0]
    assert isinstance(sample_init_event, SampleInitEvent)
    assert isinstance(sample_init_event.sample.input, str)
    if resolve_attachments is not False:
        # Content was resolved before condensation, so resolving again recovers it
        assert sample_init_event.sample.input.startswith("Hey there, hipster!")
    else:
        # Content was never resolved, so it stays as attachment refs that resolve
        assert sample_init_event.sample.input.startswith("Hey there, hipster!")

    model_event = log.samples[0].events[6]
    assert isinstance(model_event, ModelEvent)
    assert model_event.call is not None
    model_event_call_messages = model_event.call.request.get("messages")
    assert isinstance(model_event_call_messages, list)
    model_event_call_message = model_event_call_messages[0]
    assert isinstance(model_event_call_message, dict)
    model_event_call_message_content = model_event_call_message.get("content")
    assert isinstance(model_event_call_message_content, str)
    if resolve_attachments == "full":
        # Full resolve during convert means content was expanded then re-condensed
        assert model_event_call_message_content.startswith("Hey there, hipster!")
    else:
        # Core/no resolve keeps call.request content as attachment refs
        assert model_event_call_message_content.startswith("Hey there, hipster!")


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
def test_convert_applies_message_pool_dedup(
    tmp_path: pathlib.Path,
    stream: bool,
):
    """Converting a v2 .eval file should apply message pool dedup."""
    input_file = (
        _TESTS_DIR
        / "test_list_logs/2024-11-05T13-32-37-05-00_input-task_hxs4q9azL3ySGkjJirypKZ.eval"
    )

    convert_eval_logs(
        str(input_file),
        "eval",
        str(tmp_path),
        overwrite=True,
        stream=stream,
    )

    output_file = (tmp_path / input_file.name).with_suffix(".eval")
    assert output_file.exists()

    log = read_eval_log(str(output_file))
    assert log.version == LOG_SCHEMA_VERSION
    assert log.samples

    for sample in log.samples:
        model_events = [
            e
            for e in sample.events
            if isinstance(e, ModelEvent) and e.input_refs is not None
        ]
        if model_events:
            # If any model events have input_refs, message_pool must be populated
            assert len(sample.message_pool) > 0
            # input should be empty (replaced by refs)
            for me in model_events:
                assert me.input == []
                assert me.input_refs is not None
                assert len(me.input_refs) > 0
