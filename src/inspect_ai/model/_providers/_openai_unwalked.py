"""Keep the conversation out of the openai SDK's request-body walk.

`create()` runs every keyword argument through the SDK's `maybe_transform`,
which walks it node by node (~0.5 ms per conversation item on a Hawk runner).
`extra_body` is merged into the request as given, so passing the large,
already-plain-JSON fields there sends the same JSON without the walk.
"""

from typing import Any

# fields that grow with the conversation; everything else is small
_LARGE = ("input", "messages", "tools")


def unwalked(request: dict[str, Any]) -> dict[str, Any]:
    large = {k: request[k] for k in _LARGE if k in request}
    if not large:
        return request
    kwargs = {k: v for k, v in request.items() if k not in large}
    # create() requires `messages` for chat completions; extra_body overrides it
    if "messages" in large:
        kwargs["messages"] = []
    # the caller's own extra_body wins, as it did over these fields before
    kwargs["extra_body"] = {**large, **(request.get("extra_body") or {})}
    return kwargs
