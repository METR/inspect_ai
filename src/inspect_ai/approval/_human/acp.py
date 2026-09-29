"""ACP routing for human tool-call approval prompts.

When an ACP server is enabled for the running sample, the
``human_approver`` routes its prompts through ACP's
``session/request_permission`` and waits for a client to attach.
Without an ACP server, the caller retains the existing panel or
console flow.

Legacy clients receive requests exclusively, with fallback on
disconnect. When the driver opts into shared approvals, compatible
clients (including late attachments) receive the same request.
The first valid choice wins, and an explicit resolution notification
clears each participant's card. Legacy clients are never included in
that broadcast because they cannot clear a remotely resolved card.

Wait-forever: no timeout. The human at the editor is the source
of truth; default-deny on timeout would be surprising and matches
the in-proc human approver's blocking behavior.

Option round-trip: each ``PermissionOption.optionId`` is the
literal :data:`ApprovalDecision` string (``"approve"``,
``"reject"``, etc.) so the response maps back losslessly.

asyncio boundary note
=====================

This module is intentionally **asyncio-bound** (not anyio). It
awaits asyncio futures returned by ``conn.send_request`` (the
``acp`` library is asyncio-only). Cancellation catches use
``anyio.get_cancelled_exc_class()`` so they're backend-agnostic
even though the orchestration is asyncio.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Protocol, cast
from uuid import uuid4

import anyio
from acp.schema import (
    PermissionOption,
    PermissionOptionKind,
    RequestPermissionRequest,
    RequestPermissionResponse,
    ToolCallUpdate,
)

from inspect_ai.agent._acp._guards import acp_guard
from inspect_ai.tool._tool_call import (
    ToolCall,
    ToolCallView,
    substitute_tool_call_content,
)

from .._approval import Approval, ApprovalDecision

if TYPE_CHECKING:
    from acp.schema import ContentToolCallContent

    from inspect_ai.agent._acp.transport import ApproverClient, SharedApproverClient
    from inspect_ai.tool._tool_call import ToolCallContent

logger = getLogger(__name__)


class _ApprovalRoutingSession(Protocol):
    """Narrowed view of ``AcpTransport`` for the approval shim.

    Only the primitives the shim actually uses: the driver-chain
    snapshot and the attach subscription. Narrowed (rather than
    parameterising on the full
    :class:`~inspect_ai.agent._acp.transport.AcpTransport`) so tests
    can pass a minimal stub without implementing the full session
    surface.
    """

    def approver_driver_chain(self) -> list["ApproverClient"]: ...

    def subscribe_approver_attach(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]: ...


# Stable mapping from the configured ``human_approver`` choices to
# ACP ``PermissionOptionKind``. The kind is a hint to the client
# about how to visually present the option (typically the
# allow/reject color treatment); the actual decision routes via
# ``optionId``. Mappings here are best-effort semantic neighbors —
# ACP's kinds are binary allow/deny variants and Inspect's
# ``terminate`` / ``escalate`` / ``modify`` don't have perfect ACP
# counterparts.
_KIND_BY_DECISION: dict[ApprovalDecision, PermissionOptionKind] = {
    "approve": "allow_once",
    "modify": "allow_once",  # "approve with modification"
    "reject": "reject_once",
    "terminate": "reject_always",  # strongest reject — also stops the eval
    "escalate": "reject_once",  # no real ACP equivalent
}

# Display labels for each decision. Editors render these as the
# button text on the permission card.
_LABEL_BY_DECISION: dict[ApprovalDecision, str] = {
    "approve": "Approve",
    "modify": "Modify",
    "reject": "Reject",
    "terminate": "Terminate",
    "escalate": "Escalate",
}


def _options_from_choices(choices: list[ApprovalDecision]) -> list[PermissionOption]:
    """Build ACP permission options from the approver's configured choices.

    ``optionId`` is the literal decision string so the response
    round-trips back to an :class:`ApprovalDecision` without a
    lookup table. ``name`` is the human-readable button label.
    """
    return [
        PermissionOption(
            option_id=choice,
            name=_LABEL_BY_DECISION.get(choice, choice.capitalize()),
            kind=_KIND_BY_DECISION.get(choice, "reject_once"),
        )
        for choice in choices
    ]


def _approval_from_response(
    response: RequestPermissionResponse,
    choices: list[ApprovalDecision],
) -> Approval:
    """Map a client's response back to an :class:`Approval`.

    ``outcome.cancelled`` → ``Approval(decision="reject", explanation=...)``.
    ``outcome.selected`` with a recognized ``optionId`` →
    ``Approval(decision=<that decision>)``.
    ``outcome.selected`` with an unrecognized ``optionId`` → reject
    with an explanation noting the unknown id; defensive against a
    misbehaving client (or a client that synthesized its own option).
    """
    outcome = response.outcome
    # The discriminator is ``outcome.outcome`` ("selected" | "cancelled").
    if outcome.outcome == "cancelled":
        return Approval(
            decision="reject",
            explanation="ACP client cancelled the permission request.",
        )
    option_id = outcome.option_id  # AllowedOutcome
    if option_id in choices:
        # Safe cast: option_id is one of the literal-string members.
        decision: ApprovalDecision = option_id  # type: ignore[assignment]
        return Approval(decision=decision)
    return Approval(
        decision="reject",
        explanation=(
            f"ACP client returned an unknown optionId {option_id!r}; "
            f"valid options were {choices!r}."
        ),
    )


def _format_view_content_as_markdown(
    view_content: "ToolCallContent",
) -> "ToolCallContent":
    """Bake title heading + format hint into the markdown text.

    The result has ``format="markdown"`` and ``content`` that
    includes a bold title heading (if ``view_content.title`` is set)
    and — for non-markdown source — a fenced code block around the
    body so whitespace / indentation are preserved by any markdown
    renderer.

    This is what lets the inline approval section (in the TUI) and
    any other ACP client (Zed etc.) render the same visual structure
    as the in-proc ``ApprovalPanel``'s ``render_tool_approval``
    output (bold per-half titles, code fencing for plain text)
    without needing any non-standard ``_meta`` markers on the wire.
    """
    from inspect_ai.tool._tool_call import ToolCallContent

    parts: list[str] = []
    if view_content.title:
        parts.append(f"**{view_content.title}**")
        parts.append("")  # blank line between heading and body
    if view_content.format == "markdown":
        parts.append(view_content.content)
    else:
        # Fence plain text so the renderer treats it as preformatted
        # (preserves indentation; avoids markdown reinterpretation).
        # Pick a fence longer than any backtick run in the content so
        # text containing literal ``` (e.g. a viewer dumping help
        # output, or a tool printing markdown source) doesn't break
        # out of the fence and render as live markdown.
        fence = _safe_code_fence(view_content.content)
        parts.append(fence)
        parts.append(view_content.content)
        parts.append(fence)
    return ToolCallContent(
        title=None,
        format="markdown",
        content="\n".join(parts),
    )


def _safe_code_fence(content: str) -> str:
    """Return a backtick fence longer than any backtick run in ``content``.

    CommonMark / GFM rule: a fenced code block opened with N
    backticks closes at the next line whose fence has at least N
    backticks. Pick ``max_run + 1`` so the close fence we emit
    can't be matched by anything embedded in the content.

    Minimum length 3 — keeps standard plaintext (no backticks at
    all) wrapped in the familiar ``` fence.
    """
    max_run = 0
    cur_run = 0
    for ch in content:
        if ch == "`":
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return "`" * max(3, max_run + 1)


def _separator_block() -> "ContentToolCallContent":
    """A ``---`` markdown horizontal-rule block.

    Inserted between ``view.context`` and ``view.call`` when both
    are present so the rendered card mirrors the in-proc panel's
    ``Rule(characters="․")`` separator (``render_tool_approval`` in
    ``approval/_human/util.py``). Any markdown renderer draws this
    as a horizontal line; no protocol extension required.

    Body is just ``"---"`` — no surrounding newlines. Each
    ``ContentToolCallContent`` block is rendered as its own
    structural unit by every ACP client we target (each goes into
    a separate widget on our TUI, a separate Markdown block in
    Zed), so the rule is already on a line by itself. Leading /
    trailing newlines here would render as redundant blank rows
    on top of the inline section's per-block margin, doubling the
    vertical gap around the divider.
    """
    from acp.schema import ContentToolCallContent, TextContentBlock

    return ContentToolCallContent(
        type="content",
        content=TextContentBlock(type="text", text="---"),
    )


def _build_request(
    *,
    session_id: str,
    call: ToolCall,
    view: ToolCallView,
    choices: list[ApprovalDecision],
) -> RequestPermissionRequest:
    """Construct the ACP request body from Inspect's approval inputs.

    The ``tool_call`` field carries the same rich-content shape we
    already send for live tools: a descriptive title (``bash ls -la``
    rather than just ``bash``), the tool's ``ToolCallView`` content
    as inline markdown, ``raw_input`` for the debug view. Reuses
    :func:`inspect_ai.agent._acp.tool_content.descriptive_title` and
    :func:`content_blocks_from_view` so the approval prompt and the
    live tool-call rendering stay visually consistent in editors.

    Content layout — diverges from the in-proc
    ``ApprovalPanel`` / ``render_tool_approval`` in NOT including
    the model's accompanying message text. That text already streams
    to the client as a normal ``agent_message_chunk`` notification
    and renders as an assistant chip in the conversation immediately
    above the tool-call card (the approval shim's drain barrier
    guarantees it arrives BEFORE the permission request); embedding
    it AGAIN inside the approval card would duplicate the same text
    a few rows apart. The panel needs to be self-contained because
    it's a separate UI surface; the inline approval card lives in
    the transcript flow where the assistant chip is right there.

    1. **View context** (if any) — current tool state from a
       previous step.
    2. **View call** — what the agent wants to do next.

    The view halves are passed through ``substitute_tool_call_content``
    first so any ``{{param}}`` placeholders in a custom viewer
    resolve to actual argument values — prevents the editor card
    from showing literal ``{{command}}`` / ``{{path}}`` placeholders.
    """
    # Deferred imports to avoid an import cycle through
    # ``inspect_ai.agent._acp.tool_content`` → ``inspect_ai.log._transcript``
    # (which the approval module is loaded too early to participate
    # in at registry-init time). Routing through ACP only fires at
    # actual approval time, so the deferral has no perf cost.
    from inspect_ai.agent._acp.tool_content import (
        content_blocks_from_view,
        descriptive_title,
    )

    title = descriptive_title(call.function, call.arguments)

    # Substitute {{param}} placeholders in the view so the editor
    # sees concrete values, not template syntax. Mirrors
    # render_tool_approval's pre-render step.
    arguments = call.arguments or {}
    substituted_context = (
        substitute_tool_call_content(view.context, arguments)
        if view.context is not None
        else None
    )
    substituted_call = (
        substitute_tool_call_content(view.call, arguments)
        if view.call is not None
        else None
    )

    content_blocks: list[Any] = []
    # The model's accompanying message (the "why" the agent gave
    # for this tool call) is deliberately NOT included in the
    # approval request. It already flows to the client as a normal
    # ``agent_message_chunk`` notification — rendered as an
    # assistant chip in the conversation stream immediately above
    # the tool-call card. Embedding it again inside the approval
    # card just duplicated the same text a few rows apart. Diverges
    # from the in-proc ``ApprovalPanel`` (which is a separate
    # surface and needs to be self-contained) — see the comment in
    # ``_build_request`` below.
    # 1. View context (if any). Title baked into the markdown via
    # _format_view_content_as_markdown so any renderer shows it as
    # a bold heading.
    context_blocks = (
        content_blocks_from_view(_format_view_content_as_markdown(substituted_context))
        if substituted_context is not None
        else None
    )
    call_blocks = (
        content_blocks_from_view(_format_view_content_as_markdown(substituted_call))
        if substituted_call is not None
        else None
    )
    if context_blocks:
        content_blocks.extend(context_blocks)
    # Markdown rule between context and call mirrors the in-proc
    # panel's Rule separator (render_tool_approval in
    # approval/_human/util.py).
    if context_blocks and call_blocks:
        content_blocks.append(_separator_block())
    if call_blocks:
        content_blocks.extend(call_blocks)

    tool_call = ToolCallUpdate(
        tool_call_id=call.id,
        title=title,
        status="pending",
        raw_input=call.arguments,
        # ToolCallUpdate.content is typed as a union including
        # FileEditToolCallContent / TerminalToolCallContent; we
        # only build ContentToolCallContent entries (text blocks
        # wrapping the view's markdown). Cast widens for the schema
        # without changing runtime shape.
        content=cast(Any, content_blocks or None),
    )
    return RequestPermissionRequest(
        session_id=session_id,
        tool_call=tool_call,
        options=_options_from_choices(choices),
    )


async def _request_from_driver_with_fallback(
    session: _ApprovalRoutingSession,
    request: RequestPermissionRequest,
    choices: list[ApprovalDecision],
) -> Approval:
    """Route one approval and resolve every opted-in participant on exit.

    Shared clients receive a stable ID in the permission request and the
    resulting Approval metadata. Legacy clients keep exclusive routing.
    Resolution identifies the winning connection, but the ApprovalEvent
    remains the authority that the decision was applied to the tool call.
    """
    from inspect_ai.agent._acp.inspect_ext import APPROVAL_ID_META_KEY

    approval_id = str(uuid4())
    participants: dict[int, SharedApproverClient] = {}
    result: _ApprovalResult | None = None
    try:
        result = await _dispatch_permission(
            session, request, choices, approval_id, participants
        )
        if participants:
            result.approval.metadata = {
                **(result.approval.metadata or {}),
                APPROVAL_ID_META_KEY: approval_id,
            }
        await _resolve_shared_approvals(participants, request, approval_id, result)
        await anyio.lowlevel.checkpoint_if_cancelled()
        return result.approval
    except BaseException:
        # A timeout can interrupt notification after a winner is chosen.
        # Clear that provisional decision everywhere; no Approval returns.
        await _resolve_shared_approvals(participants, request, approval_id, None)
        raise


class _ApprovalResult(NamedTuple):
    approval: Approval
    client: ApproverClient


async def _resolve_shared_approvals(
    participants: dict[int, SharedApproverClient],
    request: RequestPermissionRequest,
    approval_id: str,
    result: _ApprovalResult | None,
) -> None:
    # Cleanup survives sample cancellation, while a slow/disconnected
    # peer cannot indefinitely delay the decision or sample teardown.
    with anyio.move_on_after(1, shield=result is None):
        async with anyio.create_task_group() as group:
            for client in participants.values():
                group.start_soon(
                    _resolve_shared_approval,
                    client,
                    request.session_id,
                    approval_id,
                    result,
                )


async def _resolve_shared_approval(
    client: SharedApproverClient,
    session_id: str,
    approval_id: str,
    result: _ApprovalResult | None,
) -> None:
    try:
        await client.approval_resolved(
            session_id,
            approval_id,
            result.approval.decision if result is not None else None,
            winner=result is not None and result.client is client,
        )
    except Exception as exc:
        logger.debug("ACP approval resolution failed for client %r: %s", client, exc)


async def _drain_approval_context(client: ApproverClient) -> None:
    try:
        await client.drain_notifications()
    except Exception as exc:
        logger.warning(
            "ACP approval drain_notifications failed for client %r; "
            "proceeding with request anyway: %s",
            client,
            exc,
        )


async def _request_shared_approval(
    session: _ApprovalRoutingSession,
    request: RequestPermissionRequest,
    choices: list[ApprovalDecision],
    approval_id: str,
    participants: dict[int, SharedApproverClient],
    attempted: set[int],
) -> _ApprovalResult | None:
    """First valid choice wins among ready shared clients, including late joins."""
    from inspect_ai.agent._acp.inspect_ext import APPROVAL_ID_META_KEY
    from inspect_ai.agent._acp.transport import SharedApproverClient

    request = request.model_copy(
        update={
            "field_meta": {
                **(request.field_meta or {}),
                APPROVAL_ID_META_KEY: approval_id,
            }
        }
    )
    result: _ApprovalResult | None = None
    active = 0
    changed = anyio.Event()
    unsubscribe = session.subscribe_approver_attach(lambda: changed.set())

    async def ask(client: SharedApproverClient) -> None:
        nonlocal active, result
        try:
            await _drain_approval_context(client)
            participants[id(client)] = client
            response = await client.request_permission(request)
            outcome = response.outcome
            if (
                outcome.outcome == "selected"
                and outcome.option_id in choices
                and result is None
            ):
                # No await between checking and assigning: all requests
                # run on the same event loop, so exactly one client wins.
                result = _ApprovalResult(
                    approval=_approval_from_response(response, choices), client=client
                )
        except Exception as exc:
            logger.debug("ACP shared approval failed for client %r: %s", client, exc)
        finally:
            active -= 1
            changed.set()

    try:
        async with anyio.create_task_group() as group:
            while result is None:
                changed = anyio.Event()
                for client in session.approver_driver_chain():
                    if (
                        isinstance(client, SharedApproverClient)
                        and client.supports_shared_approvals
                        and id(client) not in attempted
                    ):
                        attempted.add(id(client))
                        active += 1
                        group.start_soon(ask, client)
                if active == 0:
                    break
                await changed.wait()
            group.cancel_scope.cancel()
        return result
    finally:
        unsubscribe()


async def _dispatch_permission(
    session: _ApprovalRoutingSession,
    request: RequestPermissionRequest,
    choices: list[ApprovalDecision],
    approval_id: str,
    participants: dict[int, SharedApproverClient],
) -> _ApprovalResult:
    """Use the driver policy, waiting for a fresh attach when clients exhaust.

    Subscribe before snapshotting so an attachment cannot be lost between
    dispatch and parking. A shared driver fans out only to opted-in clients;
    a legacy driver remains exclusive until it responds or disconnects.
    Context drains before each request so the operator sees the narration.
    """
    from inspect_ai.agent._acp.transport import SharedApproverClient

    cancel_exc = anyio.get_cancelled_exc_class()
    while True:
        # Subscribe BEFORE snapshotting / dispatching so an attach
        # that lands during the dispatch attempt still sets the
        # event we wait on below. ``anyio.Event.set`` is idempotent;
        # if attach fires before ``event.wait``, the wait returns
        # immediately and we re-iterate.
        event = anyio.Event()
        unsub = session.subscribe_approver_attach(event.set)
        try:
            clients_in_order = session.approver_driver_chain()
            attempted_shared: set[int] = set()
            if clients_in_order:
                for client in clients_in_order:
                    if (
                        isinstance(client, SharedApproverClient)
                        and client.supports_shared_approvals
                    ):
                        if id(client) not in attempted_shared:
                            shared_result = await _request_shared_approval(
                                session,
                                request,
                                choices,
                                approval_id,
                                participants,
                                attempted_shared,
                            )
                            if shared_result is not None:
                                return shared_result
                        continue
                    await _drain_approval_context(client)
                    try:
                        response = await client.request_permission(request)
                    except cancel_exc:
                        raise
                    except Exception as exc:
                        # Transport failure or other client-side error.
                        # Try the next client in the fallback chain.
                        logger.debug(
                            "ACP approval request failed for client %r; "
                            "trying next: %s",
                            client,
                            exc,
                        )
                        continue
                    else:
                        # Successful dispatch — finally below unsubscribes.
                        return _ApprovalResult(
                            approval=_approval_from_response(response, choices),
                            client=client,
                        )
            # No clients attached (yet, or after they all raised).
            # Park until a fresh attach lands (or return immediately
            # if an attach raced us between subscribe and now). Under
            # exclusive routing this also covers the very first
            # interaction — we don't fall through to panel / console.
            await event.wait()
        finally:
            unsub()


async def request_human_approval_via_acp(
    *,
    message: str,
    call: ToolCall,
    view: ToolCallView,
    choices: list[ApprovalDecision],
) -> Approval | None:
    """Route a human-approval prompt through attached ACP clients.

    Returns:
        - An :class:`Approval` when at least one ACP client responded.
        - ``None`` when no ACP server or live session is available,
          or an unexpected internal error prevents routing. With ACP
          enabled, an absent or disconnected client leaves the request
          waiting for a fresh attachment rather than falling through
          to the panel or console.

    The ``message`` argument (the assistant text accompanying the
    tool call) is accepted for signature parity with the in-proc
    ``panel_approval`` / ``console_approval`` paths, but the ACP
    flow deliberately does NOT forward it on the wire — the same
    text already streams to attached clients as a normal
    ``agent_message_chunk`` notification (rendered as an assistant
    chip in the conversation above the tool-call card). The drain
    barrier in :func:`_request_from_driver_with_fallback` ensures
    the chunk lands BEFORE the permission request so the operator
    sees the "why" above the approval card. See the
    :func:`_build_request` docstring for the full rationale.

    Hard contract: never propagates a non-cancellation exception to
    the caller. This shim runs synchronously inside
    ``human_approver`` on the agent's tool-call execution path; an
    unhandled exception here would crash the tool call (and could
    crash the eval). On any internal error we log a warning and
    return ``None`` so the caller falls back to the in-proc panel.
    ``CancelledError`` propagates naturally via :func:`acp_guard`'s
    BaseException semantics — sample-level cancel still works.
    """
    del message  # accepted for signature parity; see docstring above
    with acp_guard(
        "ACP approval routing raised; falling back to in-proc approval flow"
    ):
        # Deferred imports — avoid the registry-init-time cycle through
        # the log subsystem; only fire at actual approval time.
        from inspect_ai.agent._acp.server import acp_server_accepting_clients
        from inspect_ai.log._samples import sample_active

        # Gate on whether an AcpServer is accepting external clients,
        # NOT on whether ``sample.acp_transport`` is a live transport.
        # The Live transport is opened per-sample regardless of
        # ``--acp-server`` for sub-agent isolation; only the
        # server-running flag tells us the eval is reachable from
        # outside. Without this split, the in-proc panel would never
        # see human approval requests.
        if not acp_server_accepting_clients():
            return None
        sample = sample_active()
        if sample is None or sample.acp_transport is None:
            return None
        session = sample.acp_transport
        # ``--acp-server`` is on (the gate above proved an AcpServer is
        # accepting external clients). Under exclusive routing we route
        # via ACP regardless of attach history — the in-proc panel
        # never sees this approval. See
        # ``design/acp/elicitation.md`` "Routing policy".
        request = _build_request(
            session_id=session.session_id,
            call=call,
            view=view,
            choices=choices,
        )
        # The sample is already marked as waiting on a person by
        # `human_approver`, which wraps this dispatch and the panel / console
        # fallbacks alike — the wait is a fact about the sample, not about
        # which surface happens to serve it.
        return await _request_from_driver_with_fallback(session, request, choices)
    return None
