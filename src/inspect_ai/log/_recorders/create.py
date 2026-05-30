from typing import IO, Any, Callable, cast

from inspect_ai._util.constants import LogFormat

from .eval import EvalRecorder
from .eval_sample import EvalSampleRecorder
from .json import JSONRecorder
from .recorder import Recorder

# eval.sample registered last so its directory-sniffing handles_location only
# runs for paths the suffix-based recorders don't already claim
_recorders: dict[str, type[Recorder]] = {
    "eval": EvalRecorder,
    "json": JSONRecorder,
    "eval.sample": EvalSampleRecorder,
}


def create_recorder_for_format(
    format: LogFormat, *args: Any, **kwargs: Any
) -> Recorder:
    recorder = recorder_type_for_format(format)
    return recorder(*args, **kwargs)


def recorder_type_for_format(format: LogFormat) -> type[Recorder]:
    recorder = _recorders.get(format, None)
    if recorder:
        return recorder
    else:
        raise ValueError(f"No recorder for format: {format}")


def create_recorder_for_location(location: str, log_dir: str) -> Recorder:
    recorder = recorder_type_for_location(location)
    return cast(Callable[[str], Recorder], recorder)(log_dir)


def recorder_type_for_location(location: str) -> type[Recorder]:
    for recorder in _recorders.values():
        if recorder.handles_location(location):
            return recorder

    raise ValueError(f"No recorder for location: {location}")


def recorder_type_for_bytes(log_bytes: IO[bytes]) -> type[Recorder]:
    first_bytes = log_bytes.read(4)
    log_bytes.seek(0)

    for recorder in _recorders.values():
        if recorder.handles_bytes(first_bytes):
            return recorder

    raise ValueError(f"No recorder for bytes: {first_bytes!r}")
