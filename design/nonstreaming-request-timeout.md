# Non-streaming request timeout — derive the read budget from `max_tokens`

> **Status: implemented** (`max_tokens_timeout` in
> `src/inspect_ai/_util/http_defaults{,_httpx2}.py`; applied by
> `OpenAIAPI._nonstreaming_timeout` and
> `OpenAICompatibleAPI._nonstreaming_timeout`). Originating report:
> [METR/hawk#935](https://github.com/METR/hawk/issues/935) — "model-call
> timeout defaults: 600s SDK total timeout is too low for long reasoning
> generations (esp. non-streaming OpenAI path)". This document implements
> **Step 1** of the sharpened fix approach in
> [that issue's implementation-plan comment](https://github.com/METR/hawk/issues/935#issuecomment-5427371773),
> with its route table verified against this checkout (0.3.262-7). Companion to
> [`stream-idle-timeout.md`](stream-idle-timeout.md), which covers the
> *streaming* half of the same problem space, and to the HTTP defaults layer
> in `src/inspect_ai/_util/http_defaults_httpx2.py`.

## Problem

A non-streamed OpenAI request gets one read deadline for the entire
generation. The response body arrives only when generation finishes, so the
deadline has to cover the whole thing.

That deadline is 600s (`DEFAULT_REQUEST_TIMEOUT`, mirroring the OpenAI SDK's
own default; see "What exists today"). A reasoning-heavy generation with a
large `max_tokens` — the reported case is 32k output tokens at high reasoning
effort — can legitimately run past ten minutes. The call then fails **by
construction**:

1. httpx raises a read timeout; the SDK surfaces `APITimeoutError`
   ("Request timed out.").
2. The provider still generated (and still bills) the full completion.
3. Inspect classifies the timeout as transient and retries **from scratch**,
   producing another full-length generation that is equally likely to trip
   the same deadline.

The reporting run logged roughly 34.5k such timeouts. Nothing about it is
provider pushback or infrastructure noise — the deadline is simply set below
the work the request was asked to do, and no amount of retrying fixes that.

The knobs that exist today do not address it:

- `GenerateConfig.timeout` is the **total retry budget** for a `generate()`
  call (tenacity `stop_after_delay` semantics in `model/_retry.py`), not a
  per-request deadline. Raising it grants *more* doomed retries.
- `GenerateConfig.attempt_timeout` is an `anyio` cancel scope around one
  attempt — an upper bound, not a floor. It cannot make a request wait
  longer.
- `-M client_timeout=N` does raise the read deadline, but it is a fixed
  number the user must pick per model, and picking it correctly requires
  knowing the `max_tokens` every task in the run will ask for.
- `stream_idle_timeout` only arms on streaming attempts, and by design "the
  knob cannot kill a call that produces no chunks".

So the one input that actually determines how long a generation may
legitimately take — `max_tokens` — currently has no influence on the deadline
it is measured against.

## Goals

- **A non-streaming request whose `max_tokens` implies more than the default
  budget gets a proportionally larger read deadline, by default.** No user
  action, no per-model tuning.
- **Never shorten a deadline.** The derived value is a floor-raise only. A
  request that fits inside today's budget behaves exactly as it does today,
  bit for bit.
- **Streaming attempts are untouched.** On a streaming request the SDK's
  budget is a *per-read* (per-chunk) deadline, not a total; feeding it a
  `max_tokens`-derived number changes what it means and blunts the stall
  detection `stream_idle_timeout` exists to sharpen. See "The one-line
  hazard" below — this is the single most important constraint in the design.
- **A computed *default* for `client_timeout`, not a new mechanism.** The
  value is exactly what a user would otherwise have to pass as
  `-M client_timeout=N`, computed per request from `max_tokens`. An explicit
  `-M client_timeout` therefore suppresses it outright — the user named their
  ceiling.
- **Deployer-tunable without a release**, in the established
  `INSPECT_HTTP_*` environment-variable family, including a kill switch.

Non-goals:

- **Enabling or defaulting streaming** (Steps 2 and 3 of the plan comment).
  Step 2 is a Hawk-side config change (`-M streaming=true` / `-M stream=true`
  already reach the provider); Step 3 — streaming *by default* — is a larger
  behavioural decision that should wait until `stream_idle_timeout`
  (meridianlabs-ai/inspect_ai#347) is wired through Hawk. This design is
  complementary to both and blocks neither.
- **Rescuing the Anthropic non-streaming path.** See "Anthropic is out of
  scope".
- **Bounding *actual* generation time.** The derived value is a worst-case
  ceiling computed from a ceiling (`max_tokens`), not a prediction. Killing
  slow-but-alive generations remains the job of `attempt_timeout` and the
  ctl monitoring surface.
- **Wasted-token accounting for timed-out generations** (the third bullet of
  the originating issue). Real, and separate: it touches usage recording, not
  timeouts. Track it on its own issue.

## Routes

From the plan comment, verified against this checkout:

| route | streams? | 600s ceiling applies? | fix |
|---|---|---|---|
| `anthropic/` | auto — when thinking is on **or** `max_tokens >= 8192` (`anthropic.py:1299` `auto_streaming`) | **No.** When it declines to stream, `max_tokens < 8192`, so the SDK's own `expected_time = 3600 × max_tokens / 128_000` is ≤ ~230s — comfortably inside the 600s deadline. | none needed |
| `openai/` | optional, **off by default** (`streaming` model arg; `_resolve_streaming` returns `model_stream_requested()` in auto mode) | **yes**, whenever not streaming | computed `client_timeout` default on the non-streaming branch |
| `openai-api/` | optional, **off by default** (`stream` model arg; `should_stream()` returns `False`) | **yes**, whenever not streaming | same |

The carve-outs where inspect deliberately *declines* to auto-stream are
exactly the places this fix has to work, since they cannot be rescued by
turning streaming on later:

- `service == "azure"` on a non-Responses model — the SDK's stream
  accumulator loses content-filter stop details (`_resolve_streaming`).
- `prompt_logprobs` set (`OpenAICompatibleAPI.auto_streamable`).
- `logprobs` on Together (`together.py:197` overrides `auto_streamable`);
  OpenRouter and Perplexity override it too.

## What exists today (the parts we build on)

**The effective read budget.** `OpenAIAPI._create_client` passes
`timeout=self.client_timeout if not None else NOT_GIVEN` and an
`http_client` built by `default_client_kwargs()`. With `timeout` not given,
the OpenAI SDK takes the client's timeout when it differs from httpx's own
default (`_base_client.py`: "if the user passed in a custom http client with
a non-default timeout set then we use that timeout"). So the effective
budget is `default_timeout()` — `httpx2.Timeout(600, connect=60)`, where 600
is `INSPECT_HTTP_REQUEST_TIMEOUT` and 60 is `INSPECT_HTTP_CONNECT_TIMEOUT`.
`self.client_timeout` is set from the `client_timeout` model arg, or 900 when
`service_tier == "flex"`.

**The three non-streaming call sites**, all built the same way — one
`request` dict, snapshotted into the `ModelCall`, then a branch on
batcher / streaming / plain:

| site | reached from |
|---|---|
| `openai_completions.py:119` `client.chat.completions.create(**request)` | `openai/` chat completions |
| `openai_responses.py:205` `client.responses.create(**request)` | `openai/` responses **and** `openai-api/` compatible providers, which call `generate_responses` directly |
| `openai_compatible.py:368` `self.client.chat.completions.create(**request)` | `openai-api/` compatible providers (vLLM, SGLang, Together, OpenRouter, Ollama, …) |

`OpenAICompatibleAPI` is **not** a subclass of `OpenAIAPI` — it derives from
`ModelAPI` and carries its own completions call site and its own
`client_timeout` (user-set only; no flex bump). It shares only
`generate_responses`. Its default is non-streaming
(`should_stream()` returns `False`; subclasses may override), so it has the
same exposure and the same hazard.

**The connect-deadline floor.** `_floor_connect_timeout` is registered as an
httpx request event hook by `default_client_kwargs()`. It **raises** a
connect deadline below `INSPECT_HTTP_CONNECT_TIMEOUT` and leaves a longer one
alone. It is a floor, not a cap — which is why this design must not hand the
SDK a bare float (see "Preserving connect/write/pool").

**Retry classification.** `openai_classify_retry` already treats
`APITimeoutError` as transient, and `Model._generate`'s tenacity loop retries
it. Nothing in the retry layer changes here.

**The TCP keepalive backstop.** `_default_socket_options()` sets
`SO_KEEPALIVE` with `TCP_KEEPIDLE=60`, `TCP_KEEPINTVL=60`, `TCP_KEEPCNT=5`,
so a peer that dies outright is detected in ~5 minutes regardless of the read
deadline. This matters for the cost analysis below.

## The one-line hazard

Both generate functions set `request["stream"] = True` **in the shared
request dict** before branching:

```python
if streaming:
    request["stream"] = True
    request["stream_options"] = {"include_usage": True}
...
if batcher:
    completion = await batcher.generate_for_request(request)
elif streaming:
    async with await client.chat.completions.create(**request) as stream:
        ...
else:
    completion = await client.chat.completions.create(**request)
```

Both branches call `create(**request)`. Putting the derived timeout into
`request` therefore reaches **both**, giving a streaming call a
`max_tokens`-derived *per-chunk idle budget* — precisely the failure mode the
originating issue's 2026-07-31 update warns about, and a silent weakening of
the signal `stream_idle_timeout` acts on. It would also pollute the logged
`ModelCall` request with a parameter that is not a wire parameter.

**The derived timeout is passed as a keyword at the non-streaming call site
only** — `create(**request, timeout=...)` — never merged into `request`. The
same applies verbatim to the Responses path. This is enforced by test, not
by comment (see "Tests").

## Design

### Derivation

```
derived = base_margin + max_tokens / tokens_per_second
timeout = derived  if derived > base_read_budget  else <unchanged>
```

- `max_tokens` is `GenerateConfig.max_tokens`. On OpenAI it maps to
  `max_completion_tokens` for reasoning models, which **includes reasoning
  tokens**, so it is the right ceiling for the whole generation.
- `base_margin` covers everything that is not token generation: queueing at
  the provider, TTFT, prompt processing. Default **300s**.
- `tokens_per_second` is a deliberately conservative output rate. Default
  **20.0**. Frontier reasoning models are commonly observed at 30–80 tok/s,
  so this is roughly 2–4× headroom.
- `base_read_budget` is the read deadline the request would otherwise get
  (600 by default, 900 under `service_tier=flex`, or
  `INSPECT_HTTP_REQUEST_TIMEOUT`). The comparison makes the change a strict
  floor-raise.

Worked values with the proposed defaults:

| `max_tokens` | derived | applied? | effective deadline |
|---|---|---|---|
| unset | — | no | 600 |
| 4,096 | 505 | no (below 600) | 600 |
| 8,192 | 710 | yes | 710 |
| 16,384 | 1,119 | yes | ~19 min |
| 32,768 | 1,938 | yes | ~32 min |
| 64,000 | 3,500 | yes | ~58 min |

Derivation engages above roughly 6k `max_tokens`.

### When `max_tokens` is unset

**No derivation.** This is a deliberate v1 restriction. With `max_tokens`
unset the model's own maximum applies (128k for the gpt-5 line), and
`ModelInfo.output_tokens` would let us derive from it — but that would move
the default deadline for *every* unset-`max_tokens` gpt-5 call to over 90
minutes, which is a far larger change than the reported problem justifies.
The originating issue's acceptance criterion is about eval sets that *do*
request large `max_tokens`, and those are covered.

Deriving from `ModelInfo.output_tokens` is a reasonable follow-up once the
`max_tokens`-set case has run in production; it should be its own change.

### Where it applies

| path | derived timeout | why |
|---|---|---|
| chat completions, non-streaming | **applied** | the reported failure |
| responses, non-streaming | **applied** | same shape, same exposure |
| chat completions / responses, streaming | **never** | budget is per-chunk; see the hazard above |
| batch (`batcher is not None`) | never | no long-lived request to bound |
| responses `background=True` | inert | `create` returns immediately; the wait is in `wait_for_background_response` polling. Not special-cased — it simply never matters. |

All three call sites are covered. The `openai-api/` compatible providers
are included deliberately: same non-streaming default, same exposure, same
one-line hazard, and their `client_timeout` precedence is simpler (no flex
bump to distinguish from a user-set value).

### Precedence

Because the derived value *is* a default for `client_timeout`:

1. `-M client_timeout=N` set explicitly → **no derivation**. The user named a
   ceiling; honour it. (Requires distinguishing a user-set `client_timeout`
   from the automatic `service_tier=flex` bump, which currently share
   `OpenAIAPI.client_timeout`. `OpenAICompatibleAPI.client_timeout` is
   user-set only, so no such split is needed there.)
2. Otherwise the effective read budget — `INSPECT_HTTP_REQUEST_TIMEOUT`, the
   flex bump, or 600 — is the **floor**, and derivation may raise above it.
3. `INSPECT_HTTP_OUTPUT_TOKENS_PER_SECOND=0` → derivation disabled entirely
   (kill switch).

### Preserving connect/write/pool

The SDK stamps a per-request timeout over the client's, and httpx expands a
bare float to all four phases. Handing it `1938.0` would make **connect**
1938s as well, and `_floor_connect_timeout` only ever raises a connect
deadline — it would leave that one alone. A black-holed SYN would then hang
for half an hour.

So the derived value is passed as an `httpx2.Timeout` built from the client's
existing one with **only `read` replaced**:

```python
httpx2.Timeout(
    connect=base.connect, read=derived, write=base.write, pool=base.pool
)
```

### Placement

A helper alongside `default_timeout()` / `connect_timeout()` in
`src/inspect_ai/_util/http_defaults_httpx2.py` — that module already owns
`DEFAULT_REQUEST_TIMEOUT`, the `INSPECT_HTTP_*` family, and the `_env_float`
validation used for all of them. It returns the `httpx2.Timeout` or `None`
for "leave unchanged".

Policy (precedence, streaming, batching) lives in `OpenAIAPI`, which is the
only object that knows whether `client_timeout` was user-set. It computes the
value once per generate and hands it to `generate_completions` /
`generate_responses` as a new keyword-only parameter defaulting to
`NOT_GIVEN`, which those functions forward **only** on the non-streaming
`create` call.

## Costs and how they are bounded

**A wedged request sits longer.** The deadline is the last defence against a
connection that stays open but never delivers. Raising it from 600s to
~1900s means a black-holing intermediary is tolerated three times as long.
Bounding that:

- A peer that dies outright is caught by TCP keepalive in ~5 minutes,
  independent of the read deadline.
- `attempt_timeout` remains available as a hard per-attempt cap for anyone
  who wants one, and is unaffected by this change.
- Only requests that asked for a large `max_tokens` are affected at all.

The asymmetry is what makes this safe as a **default-on** change, unlike
`stream_idle_timeout`, which shipped opt-in: a wrong value here cannot kill a
healthy call, only delay the failure of an already-broken one.

**SDK-internal retry amplification.** The OpenAI SDK's own
`max_retries` defaults to 2 and Inspect does not override it, so an
`APITimeoutError` costs up to 3 × the deadline before Inspect's retry loop
even sees it. That is 30 minutes today and ~97 minutes at a 1938s deadline.
This is pre-existing and orthogonal, but the multiplier grows with this
change, so it must be documented alongside the `-M max_retries=0` lever.
(Whether Inspect should set the SDK's `max_retries` to 0 and own retrying
itself is a separate question worth raising — Inspect's retry loop is
strictly better instrumented.)

**Interaction with `GenerateConfig.timeout`.** A retry budget below the
derived deadline stops retries mid-request. Both are unset by default, so
this only affects users who set `--timeout`, and the resulting behaviour
("the retry budget is what it says") is correct. Worth a docs sentence.

## Anthropic needs no fix

Per the route table: `anthropic/` auto-streams whenever thinking is on or
`max_tokens >= 8192`, so a non-streaming Anthropic call always has
`max_tokens < 8192` and an SDK-expected time of ≤ ~230s — well inside the
600s deadline. There is no exposure to close.

Two supporting facts, in case the auto-streaming rule ever changes:

- The originating issue's 2026-07-31 update is explicit that the Anthropic
  timeout storm first blamed on this issue was a different failure, and that
  a `max_tokens`-derived default "would not have prevented that incident".
- Above ~21,333 `max_tokens` the SDK raises `ValueError("Streaming is
  strongly recommended for operations that may take longer than 10
  minutes")` on any non-streaming request (upstream issue #1407), and
  inspect's provider deliberately preserves the SDK's exact `DEFAULT_TIMEOUT`
  object so that guard keeps firing (`anthropic.py`, `_http_default_args`).
  Passing a timeout there would bypass a guard mirroring a real server-side
  limit.

Google likewise applies `config.timeout` as its own overall budget and is
untouched.

## Tests

In `tests/model/providers/test_openai.py`, `test_openai_responses.py` and
`test_openai_compatible.py` (one per call site), using the established
`MagicMock` client / `AsyncMock(create)` harness. The helper itself is
covered in `tests/model/test_http_defaults.py`, which also enforces that both
flavour modules export the same names — so the httpx and httpx2 copies must
both carry it:

1. **Applied when non-streaming and large.** `max_tokens=32000` →
   `create.call_args.kwargs["timeout"]` is an `httpx2.Timeout` whose `read`
   matches the formula and whose `connect` equals `connect_timeout()`.
2. **The hazard, asserted directly.** Same config with `streaming=True` →
   `"timeout" not in create.call_args.kwargs` **and** `"timeout" not in`
   the logged `ModelCall` request. This test is the point of the exercise;
   it is what stops a future refactor from moving the keyword into the
   shared dict.
3. **Never lowers.** `max_tokens=4096` → no `timeout` keyword; the request
   is byte-identical to today's.
4. **Unset `max_tokens`** → no `timeout` keyword.
5. **Explicit `client_timeout` wins** → no `timeout` keyword even at
   `max_tokens=32000`; **flex does not suppress derivation** → applied, with
   900 as the floor.
6. **Batch path unaffected** → the batcher receives a request dict with no
   `timeout` key.
7. **Kill switch** — `INSPECT_HTTP_OUTPUT_TOKENS_PER_SECOND=0` → no
   `timeout` keyword.
8. **The helper itself**: formula, floor comparison, a base with no deadline
   left alone, env overrides, junk env values falling back, zero rate.

All three call sites get 1–3 at minimum.

## Alternatives considered

**Stream by default on the OpenAI provider** — Step 3 of the plan comment,
and the more complete fix: streaming turns a total-time failure into an idle
failure that `stream_idle_timeout` already handles. Deferred there and here
for the same reasons: it needs `stream_idle_timeout` available downstream
first, and it cannot cover the carve-outs listed under "Routes" (Azure
non-Responses, `prompt_logprobs`, Together `logprobs`), which keep needing a
timeout regardless. This change is cheap, orthogonal, and lands first.

**Derive from the measured throughput registry** (`model/_throughput.py`,
`design/model-throughput.md`). Tempting — real tok/s per model — but wrong
unit: that registry measures *aggregate* output tokens/sec across all
concurrent calls, which is many times any single call's rate, so it would
produce deadlines far too short. It is also empty at cold start and varies
run to run, making the deadline non-deterministic. Rejected.

**Change `GenerateConfig.timeout` semantics or add a `request_timeout`
field.** `timeout` is the retry budget by long-settled decision (upstream
#2568, resolved by adding `attempt_timeout` rather than changing `timeout`),
and `attempt_timeout` already occupies the per-attempt-cap slot. A third
user-facing field for "per-request floor" is more surface than the problem
needs when the value can be derived. Revisit only if the derived default
proves unfit somewhere.

**Compute the default in Hawk instead of inspect.** Rejected in the plan
comment, and the reason holds at this checkout: Hawk would have to predict
whether a call streams from the outside, shadowing three private methods
(`_resolve_streaming`, `resolve_stream`, `auto_streamable`) that changed
shape in 0.3.261 and carry no stability guarantee. Inspect is where the
streaming decision is made, so it is where the branch-dependent default
belongs.

**Just raise `DEFAULT_REQUEST_TIMEOUT`.** One number cannot be both large
enough for a 32k reasoning generation and small enough to catch a wedged 500-
token call. Deriving from `max_tokens` is the whole point.

## Open questions

1. **The defaults.** `tokens_per_second = 20.0` and `base_margin = 300s` are
   proposed, not measured. If there is per-model output-rate data from the
   reporting run, calibrate against its slowest observed generation before
   landing.
2. **Environment variable names.** `INSPECT_HTTP_OUTPUT_TOKENS_PER_SECOND`
   and `INSPECT_HTTP_TIMEOUT_MARGIN` fit the existing family but are only
   loosely "HTTP" settings. `INSPECT_MODEL_*` reads better and starts a new
   family.
3. **Should Inspect set the OpenAI SDK's `max_retries=0`?** Out of scope
   here, but this change makes the existing 3× amplification more expensive.

## Landing

Per the plan comment, Step 1 lands in **METR's inspect fork**
(`github.com/METR/inspect_ai`) with METR/hawk#935 as the motivation; if METR
likes the approach it then goes upstream. The upstream roster gate in
`AGENTS.md` governs `UKGovernmentBEIS/inspect_ai` only and does not apply to
the fork PR.

Base branch: **`metr/hotfix-0.3.261`** (tip `0b9856a`, 2026-09-03) — one
commit ahead of Hawk's pinned `98c9d82`, and the fork's live integration
line. Fork `main` is stale (2026-02-27, 3326 commits behind upstream). All
three call sites in this spec exist on that branch at effectively the same
lines as this checkout (0.3.262-7), so nothing here needs re-derivation for
the older base.

Write the change upstream-clean — no METR-specific behaviour, defaults
tunable by environment variable — so the later upstream PR is a cherry-pick
rather than a rewrite.

Sequence:

1. Implement on `metr/hotfix-0.3.261`, with the tests below.
2. `make check` + `make test`; run the async tests with `--runtrio`; one
   `/code-review` pass in a fresh context on a frontier model.
3. Upstream PR. Note that per `AGENTS.md` the operating account (`jrhender`,
   id 4407464) is **not** on the `.github/qualified.yml` roster, so the
   upstream PR needs an issue labelled `accepted` by a maintainer first —
   file it with the evidence from METR/hawk#935 (the 34.5k
   `"Request timed out."` timeouts; the `max_tokens`/deadline mismatch) and
   this design.
4. CHANGELOG, one line under `## Unreleased`, outcome not mechanism: *"Long
   non-streaming OpenAI generations with a large `max_tokens` no longer time
   out at 10 minutes."*
5. Docs: a row per new variable in the `HTTP Client Settings` table in
   `docs/models-concurrency.qmd`, plus a short paragraph covering the
   derivation, the streaming carve-out, and the SDK retry multiplier. The
   `client_timeout` paragraph in `docs/providers.qmd` should mention that it
   now suppresses the computed default.
