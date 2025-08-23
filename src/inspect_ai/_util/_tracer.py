import collections
import time
from contextlib import contextmanager
from typing import Any

import pydantic

counter: dict[tuple[str, str | None], float] | None = None


@contextmanager
def init_counter():
    global counter
    counter = collections.defaultdict(float)
    yield counter
    counter = None


class InspectBaseModel(pydantic.BaseModel):
    ...

    @pydantic.field_validator("*", mode="wrap")
    @classmethod
    def trace_validation(
        cls,
        value: Any,
        handler: pydantic.ValidatorFunctionWrapHandler,
        info: pydantic.ValidationInfo,
    ):
        if counter is None:
            return handler(value)

        start_time = time.perf_counter()
        result = handler(value)
        counter[(cls.__name__, info.field_name)] += time.perf_counter() - start_time

        return result
