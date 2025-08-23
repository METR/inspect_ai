import pathlib

from inspect_ai.log import read_eval_log
from inspect_ai.log._transcript import SampleInitEvent
from inspect_ai.model._model_output import ModelOutput


def test_lazy_validation() -> None:
    log = read_eval_log(
        pathlib.Path(__file__).parent
        / "log/test_list_logs/2024-11-05T13-32-37-05-00_input-task_hxs4q9azL3ySGkjJirypKZ.eval"
    )
    assert log.samples is not None
    assert log.samples[0].events is not None

    first_event_raw = log.samples[0].events.root[0]
    assert isinstance(first_event_raw, dict)
    first_event = log.samples[0].events[0]
    assert isinstance(first_event, SampleInitEvent)
    model_event = next(
        event for event in log.samples[0].events if event.event == "model"
    )
    output_raw = model_event._output  # type: ignore
    assert model_event._output_computed is False  # type: ignore
    assert output_raw is None

    model_output = model_event.output
    assert isinstance(model_output, ModelOutput)
    assert model_event._output_computed is True  # type: ignore
