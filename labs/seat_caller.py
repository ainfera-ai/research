"""AIN-542 Step 4 · live cross-family seat caller (gateway, OpenAI-compatible).

The seam that turns the pure Council into live verdicts. `gateway_seat_caller`
returns a `SeatCaller` that, given a seat + two candidate outputs in order, asks
that seat's model (via the ainfera gateway) which response is better and maps the
reply to `'first' | 'second' | 'tie'`. Position-randomisation + family-exclusion
+ aggregation happen in `labs.council`; this only does the one call.

Spark optimisation: open-weight seats (`seat.on_spark`) run on Spark Labs; pass a
client pointed at the local Spark endpoint for those and the gateway for frontier
seats. The client is injected (OpenAI-compatible) so this stays testable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from labs.council_seats import Seat

SeatCaller = Callable[[Seat, str, str], str]

PAIRWISE_SYSTEM = (
    "You are an impartial evaluator. You are shown a task and two candidate "
    "responses, FIRST and SECOND. Decide which response is better (more correct, "
    "complete, and useful). Reply with exactly one word: FIRST, SECOND, or TIE. "
    "No explanation."
)


def build_pairwise_messages(first: str, second: str) -> list[dict[str, str]]:
    user = (
        f"FIRST response:\n{first}\n\n"
        f"SECOND response:\n{second}\n\n"
        "Which is better — FIRST, SECOND, or TIE?"
    )
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {"role": "user", "content": user},
    ]


_FIRST = re.compile(r"\bfirst\b|\b1\b|\boption a\b|\bresponse a\b|^a\b", re.IGNORECASE)
_SECOND = re.compile(
    r"\bsecond\b|\b2\b|\boption b\b|\bresponse b\b|^b\b", re.IGNORECASE
)


def parse_pick(text: str | None) -> str:
    """Map a seat's reply to 'first'|'second'|'tie'. Unparseable → 'tie' (so a
    confused seat doesn't inject a spurious preference)."""
    if not text:
        return "tie"
    t = text.strip().lower()
    if "tie" in t or "equal" in t or "both" in t:
        return "tie"
    f, s = bool(_FIRST.search(t)), bool(_SECOND.search(t))
    if f and not s:
        return "first"
    if s and not f:
        return "second"
    return "tie"


def gateway_seat_caller(
    client: Any, *, max_tokens: int = 8, temperature: float = 0.0
) -> SeatCaller:
    """Build a SeatCaller over an OpenAI-compatible `client` (e.g.
    `openai.OpenAI(base_url=AINFERA_BASE_URL, api_key=...)`). On any error the
    seat abstains (returns 'tie') so one flaky seat never crashes a verdict."""

    def call(seat: Seat, first: str, second: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=seat.model_slug,
                messages=build_pairwise_messages(first, second),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return parse_pick(resp.choices[0].message.content)
        except Exception:  # noqa: BLE001 — a flaky seat abstains, never crashes
            return "tie"

    return call


# slug → maker family, for mapping candidate-output models to the self-preference
# exclusion key. Substring match, most-specific first; unknown → 'unknown'.
_FAMILY_BY_SUBSTR: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("qwen", "alibaba"),
    ("deepseek", "deepseek"),
    ("minimax", "minimax"),
    ("glm", "zai"),
    ("nemotron", "nvidia"),
    ("mimo", "xiaomi"),
    ("phi", "microsoft"),
    ("ernie", "baidu"),
)


def family_of_slug(slug: str | None) -> str:
    s = (slug or "").lower()
    for sub, fam in _FAMILY_BY_SUBSTR:
        if sub in s:
            return fam
    return "unknown"
