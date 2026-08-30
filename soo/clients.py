"""
soo/clients.py — async chat layer over Anthropic, OpenAI, Bedrock, plus a stub.

One coroutine, `chat()`, fronts every provider so callers never branch on
vendor. Concurrency, backoff and error classification mirror the pattern in
core/rewards/rewards/llm_judge.py (semaphore + gather + exponential backoff,
retry only transient classes).

Design notes worth knowing before editing:

* No sampling parameters are sent anywhere. Claude Sonnet 5 rejects a
  non-default ``temperature`` / ``top_p`` / ``top_k`` with a 400, and running
  every model at its provider default is what real users get.
* OpenAI reasoning models (gpt-5.x) take ``max_completion_tokens``, not
  ``max_tokens``; sending the latter is an error.
* A failed call returns a record with ``error`` set rather than raising. The
  grid is worth more partially complete than not at all, and resume will retry
  the gaps on the next run.
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
from typing import Any

from anthropic import AsyncAnthropic
import boto3
from openai import AsyncOpenAI

from .config import (
    ANTHROPIC_API_KEY,
    AWS_PROFILE,
    AWS_REGION,
    BEDROCK_CANDIDATES,
    MAX_RETRIES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REQUEST_TIMEOUT_S,
    RETRY_BASE_DELAY_S,
)


# ---------------------------------------------------------------------------
# Lazy client singletons — constructed on first use so that importing this
# module never requires credentials (tests and --dry-run must work offline).
# ---------------------------------------------------------------------------
_anthropic_client: Any = None
_openai_client: Any = None
_bedrock_client: Any = None
_resolved_bedrock_model: str | None = None

def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:

        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=REQUEST_TIMEOUT_S)
    return _anthropic_client

def _get_openai():
    global _openai_client
    if _openai_client is None:

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=REQUEST_TIMEOUT_S)
    return _openai_client

def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:

        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        _bedrock_client = session.client("bedrock-runtime")
    return _bedrock_client

def resolve_bedrock_model(candidates: list[str] | None = None) -> str:
    """Pick the first BEDROCK_CANDIDATES entry the account can actually invoke.

    Account enablement varies, so the preference order is resolved against the
    live foundation-model list rather than assumed. Cached after the first call.

    Raises RuntimeError if none are available — callers should surface that to
    the user rather than silently substituting a different model, because which
    model represents the third family is a research decision, not a detail.
    """
    global _resolved_bedrock_model
    if _resolved_bedrock_model is not None:
        return _resolved_bedrock_model

    candidates = candidates or BEDROCK_CANDIDATES
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    control = session.client("bedrock")
    listing = control.list_foundation_models()

    # An inference profile ID ("eu.amazon.nova-pro-v1:0") is not itself a
    # foundation-model ID ("amazon.nova-pro-v1:0"), so match on the suffix.
    available: set[str] = set()
    for summary in listing.get("modelSummaries", []):
        model_id = summary.get("modelId", "")
        available.add(model_id)

    for candidate in candidates:
        bare = candidate.split(".", 1)[1] if candidate.startswith(("eu.", "us.")) else candidate
        if any(bare == m or m.startswith(bare.split(":")[0]) for m in available):
            _resolved_bedrock_model = candidate
            print(f"[clients] bedrock family resolved to {candidate}", file=sys.stderr)
            return candidate

    raise RuntimeError(
        "none of the Bedrock candidates are available on this account: "
        + ", ".join(candidates)
        + f" (region {AWS_REGION}, profile {AWS_PROFILE})"
    )

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
_RETRYABLE_TOKENS = (
    "rate limit",
    "rate_limit",
    "429",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "throttl",
    "service unavailable",
    "internal server error",
    "500",
    "502",
    "503",
    "529",
)

def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only; a 400 is a malformed request, not bad luck."""
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "timeout" in name or "connection" in name or "overloaded" in name:
        return True
    if "badrequest" in name or "notfound" in name or "permissiondenied" in name or "authentication" in name:
        return False
    text = str(exc).lower()
    return any(token in text for token in _RETRYABLE_TOKENS)

async def _with_retry(coro_factory, label: str) -> Any:
    """Run an async factory with exponential backoff and jitter."""
    last: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - classified below
            last = exc
            if not _is_retryable(exc):
                raise
            if attempt == MAX_RETRIES - 1:
                break
            delay = RETRY_BASE_DELAY_S * (2**attempt) + random.uniform(0, 1)
            print(f"[clients] {label} retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s ({exc})", file=sys.stderr)
            await asyncio.sleep(delay)
    raise last if last else RuntimeError(f"{label} failed with no exception recorded")

# ---------------------------------------------------------------------------
# Per-provider calls. Each returns (text, stop_reason, usage_dict).
# ---------------------------------------------------------------------------
async def _chat_anthropic(
    model_id: str,
    messages: list[dict],
    system: str | None,
    max_tokens: int,
    json_schema: dict | None,
    disable_thinking: bool,
) -> tuple[str, str, dict]:
    client = _get_anthropic()
    kwargs: dict[str, Any] = {"model": model_id, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    if disable_thinking:
        # Judge path: cheaper, lower-variance, and no reasoning tokens to pay for.
        kwargs["thinking"] = {"type": "disabled"}
    if json_schema is not None:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

    response = await client.messages.create(**kwargs)
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, str(response.stop_reason), usage

async def _chat_openai(
    model_id: str,
    messages: list[dict],
    system: str | None,
    max_tokens: int,
    json_schema: dict | None,
) -> tuple[str, str, dict]:
    client = _get_openai()
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": full_messages,
        # gpt-5.x are reasoning models: max_completion_tokens, not max_tokens.
        "max_completion_tokens": max_tokens,
    }
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "rubric", "strict": True, "schema": json_schema},
        }

    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return text, str(choice.finish_reason), usage

async def _chat_bedrock(
    model_id: str,
    messages: list[dict],
    system: str | None,
    max_tokens: int,
) -> tuple[str, str, dict]:
    """Bedrock Converse — model-agnostic, so one path covers Nova/Qwen/DeepSeek.

    boto3 is synchronous, so the call runs in a worker thread to keep the
    asyncio gather non-blocking.
    """
    client = _get_bedrock()
    converse_messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    response = await asyncio.to_thread(client.converse, **kwargs)
    blocks = response["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks)
    usage_raw = response.get("usage", {})
    usage = {
        "input_tokens": usage_raw.get("inputTokens", 0),
        "output_tokens": usage_raw.get("outputTokens", 0),
    }
    return text, str(response.get("stopReason", "")), usage

def _stub_text(messages: list[dict], json_schema: dict | None) -> str:
    """Deterministic canned output for --dry-run: exercises the pipeline with no API calls.

    When a schema is present the stub emits a schema-shaped object so the
    judging and analysis paths exercise real parsing rather than a special case.
    """
    if json_schema is not None:
        props = json_schema.get("properties", {})
        stub: dict[str, Any] = {}
        for key, spec in props.items():
            if spec.get("type") == "integer":
                lo = spec.get("minimum", 0)
                stub[key] = lo
            elif spec.get("type") == "string":
                enum = spec.get("enum")
                stub[key] = enum[0] if enum else "stub"
            else:
                stub[key] = 0
        return json.dumps(stub)

    last = messages[-1]["content"] if messages else ""
    return (
        "[DRY RUN] This is stub output, not a model response. "
        "It exists so the pipeline can be exercised end to end without spending anything. "
        f"The final user turn was {len(last)} characters long."
    )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def chat(
    provider: str,
    model_id: str | None,
    messages: list[dict],
    *,
    max_tokens: int,
    system: str | None = None,
    json_schema: dict | None = None,
    disable_thinking: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    label: str = "chat",
) -> dict:
    """Call one model and return a plain dict.

    Never raises for API failures — a failed call comes back with ``error`` set
    and empty ``text`` so one bad row cannot abort a 4,000-call grid. Resume
    picks the gap up next run.
    """
    async def _run() -> tuple[str, str, dict]:
        if provider == "stub":
            await asyncio.sleep(0)
            return _stub_text(messages, json_schema), "end_turn", {"input_tokens": 0, "output_tokens": 0}
        if provider == "anthropic":
            return await _chat_anthropic(model_id, messages, system, max_tokens, json_schema, disable_thinking)
        if provider == "openai":
            return await _chat_openai(model_id, messages, system, max_tokens, json_schema)
        if provider == "bedrock":
            return await _chat_bedrock(model_id, messages, system, max_tokens)
        raise ValueError(f"unknown provider: {provider}")

    async def _guarded() -> dict:
        try:
            text, stop_reason, usage = await _with_retry(_run, label)
            return {
                "text": text,
                "stop_reason": stop_reason,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, record the fault
            print(f"[clients] {label} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return {
                "text": "",
                "stop_reason": "error",
                "input_tokens": 0,
                "output_tokens": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if semaphore is None:
        return await _guarded()
    async with semaphore:
        return await _guarded()

async def gather_chats(tasks: list[dict], max_concurrency: int) -> list[dict]:
    """Run many `chat` calls concurrently under one semaphore.

    Each task dict is forwarded to `chat` as keyword arguments.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    return await asyncio.gather(*(chat(semaphore=semaphore, **task) for task in tasks))
