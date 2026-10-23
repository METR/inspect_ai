from typing import Literal

from pydantic import BaseModel


# Shapes for peeking at host->injectable control RPCs — a different wire
# contract from the server-relay envelope models in
# _remote_tools/_mcp/jsonrpc_types.py; the two are not interchangeable.
class JSONRPCIncoming(BaseModel):
    jsonrpc: Literal["2.0"]
    method: str
    params: list[object] | dict[str, object] | None = None


class JSONRPCRequest(JSONRPCIncoming):
    id: int | float | str


class JSONRPCNotification(JSONRPCIncoming):
    pass
