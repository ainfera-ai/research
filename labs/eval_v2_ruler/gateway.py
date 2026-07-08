"""gateway.py — the single seam to the Ainfera inference gateway.

Calls models via the OpenAI-compat shim (POST /v1/chat/completions) with
tools[]. The router translates tools through (AIN-347 retired the blanket
422; tools_dropped is always False now).

Pinned model slugs write NO routing_outcomes row — excluded from training
by construction. This is NOT ainfera-inference auto-route.

Pure stdlib (urllib) — no SDK dependency. Mockable for tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from labs.eval_v2_ruler import config


@dataclass(frozen=True)
class CallResult:
    """One inference call result."""

    content: str
    tool_calls: list[dict[str, Any]]  # [{name, arguments(dict|None), args_raw}]
    model_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    tools_dropped: bool
    error: str | None  # non-None if the call failed after retries


class GatewayClient:
    """Wraps the Ainfera gateway OpenAI-compat endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.base_url = base_url or config.GATEWAY_BASE_URL
        self._api_key = api_key or os.environ.get(config.GATEWAY_KEY_ENV)
        self.agent_id = agent_id or os.environ.get(config.AGENT_ID_ENV, config.DEFAULT_AGENT_ID)
        if not self._api_key:
            raise GatewayError(f"no {config.GATEWAY_KEY_ENV} set — live calls need the fleet key.")

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> CallResult:
        """One inference call with retry. Returns a CallResult (never raises
        on transient failure — returns error= instead, so the scorer can
        record a 0 and move on)."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or config.MAX_TOKENS,
            "temperature": 0,
            "agent_id": self.agent_id,
        }
        if tools:
            body["tools"] = tools

        last_err: str | None = None
        for attempt in range(config.RETRIES + 1):
            try:
                return self._do_call(body)
            except urllib.error.HTTPError as e:
                # 4xx = don't retry (bad request, auth, etc.)
                if 400 <= e.code < 500 and e.code != 429:
                    return CallResult(
                        content="", tool_calls=[], model_used=model,
                        latency_ms=0, input_tokens=0, output_tokens=0,
                        finish_reason=None, tools_dropped=False,
                        error=f"HTTP {e.code}: {e.read().decode()[:200]}",
                    )
                last_err = f"HTTP {e.code}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:150]}"

            if attempt < config.RETRIES:
                time.sleep(config.BACKOFF_BASE * (2**attempt))

        return CallResult(
            content="", tool_calls=[], model_used=model,
            latency_ms=0, input_tokens=0, output_tokens=0,
            finish_reason=None, tools_dropped=False,
            error=last_err or "unknown error",
        )

    def _do_call(self, body: dict[str, Any]) -> CallResult:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "eval-v2-ruler/0.1",
            },
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=config.CALL_TIMEOUT) as r:
            dt = time.monotonic() - t0
            raw = json.loads(r.read())

        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        raw_tool_calls = msg.get("tool_calls") or []

        tool_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            parsed_args = None
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except (ValueError, TypeError):
                    parsed_args = None
            elif isinstance(args, dict):
                parsed_args = args
            tool_calls.append({
                "name": fn.get("name"),
                "arguments": parsed_args,
                "args_raw": args,
            })

        usage = raw.get("usage") or {}
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        tools_dropped = hdrs.get("x-ainfera-tools-dropped") == "true"

        return CallResult(
            content=content,
            tool_calls=tool_calls,
            model_used=raw.get("model", body["model"]),
            latency_ms=round(dt * 1000),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason"),
            tools_dropped=tools_dropped,
            error=None,
        )


class GatewayError(RuntimeError):
    """Raised for misconfiguration at the seam."""
