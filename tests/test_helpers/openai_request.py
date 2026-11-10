from collections.abc import Mapping
from typing import Any

from openai import NotGiven, Omit


def sent_request(create_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The body an openai SDK create() call sends: its kwargs with extra_body merged in, unset values dropped."""
    kwargs = dict(create_kwargs)
    extra_body = kwargs.pop("extra_body", None) or {}
    return {
        k: v
        for k, v in {**kwargs, **extra_body}.items()
        if not isinstance(v, (NotGiven, Omit))
    }
