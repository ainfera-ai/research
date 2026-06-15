"""gateway.py — the single seam to the Ainfera inference gateway.

EVERY live model call goes through here. The `ainfera` SDK is **lazy-imported**
inside `_agent()` so this module (and the unit tests, which mock this seam) need
no SDK installed; CI stays SDK-free.

Contract (api InferenceRequest/InferenceResponse, verified 2026-06-15):
  POST {AINFERA_BASE_URL}/inference   Authorization: Bearer {AINFERA_API_KEY}
  body: {agent_id, model, messages, max_tokens, temperature?, routing_hint?, task_type?}
  resp: {content, cost_usd, input_tokens, output_tokens, model_used, inference_id, ...}
  - model="<slug>"            → pinned passthrough; NO routing_outcomes row (excluded).
  - model="ainfera-inference" → brain route; writes a row → MUST be a probe agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from labs.eval_harness import config

ROUTED_MODEL = config.ROUTED_MODEL


@dataclass(frozen=True)
class GatewayResult:
    content: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    model_used: str
    inference_id: str | None = None


class GatewayError(RuntimeError):
    """Raised for any misconfiguration or transport failure at the seam."""


class GatewayClient:
    """Wraps the `ainfera` SDK. Constructs lazily; refuses to call without a key."""

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.agent_id = agent_id or os.environ.get(config.PROBE_AGENT_ID_ENV)
        self.base_url = base_url or os.environ.get(
            config.GATEWAY_BASE_URL_ENV, "https://api.ainfera.ai/v1"
        )
        self._api_key = api_key or os.environ.get(config.GATEWAY_KEY_ENV)
        self._sdk_agent: Any = None

    def _agent(self) -> Any:
        if self._sdk_agent is None:
            if not self._api_key:
                raise GatewayError(
                    f"no {config.GATEWAY_KEY_ENV} set — live calls need the Labs key (Doppler)."
                )
            try:
                from ainfera import AinferaClient  # lazy: keeps tests + CI SDK-free
            except ImportError as e:  # pragma: no cover - exercised only on the Labs runner
                raise GatewayError(
                    "the `ainfera` SDK is not installed — `pip install ainfera` on the Labs runner."
                ) from e
            client = AinferaClient(
                api_key=self._api_key, base_url=self.base_url.removesuffix("/v1")
            )
            self._sdk_agent = client.agents.retrieve(agent_id=self.agent_id)
        return self._sdk_agent

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        task_type: str | None = None,
        routing_hint: dict[str, Any] | None = None,
    ) -> GatewayResult:
        """One inference call. `model` pins a slug, or ROUTED_MODEL to auto-route."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or config.MAX_TOKENS,
        }
        if task_type is not None:
            kwargs["task_type"] = task_type
        if routing_hint is not None:
            kwargs["routing_hint"] = routing_hint
        resp = self._agent().inference(**kwargs)
        return GatewayResult(
            content=resp.content,
            cost_usd=float(resp.cost_usd),
            input_tokens=int(resp.input_tokens),
            output_tokens=int(resp.output_tokens),
            model_used=resp.model_used,
            inference_id=str(getattr(resp, "inference_id", "")) or None,
        )
