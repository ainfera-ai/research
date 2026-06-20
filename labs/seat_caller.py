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
import time
from typing import Any

from labs.council_seats import Seat, family_of

SeatCaller = Callable[[Seat, str, str], str]

# Canonical slug→family map lives in council_seats now (AIN-546 single source of
# truth). Re-exported here for back-compat with existing call sites.
family_of_slug = family_of

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


def _complete_with_retry(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    retries: int,
    backoff_base: float,
) -> Any:
    """One completion with retry + exponential backoff on transient errors (the
    gateway 502s seen live). Raises the last error once retries are exhausted."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries and backoff_base > 0:
                time.sleep(backoff_base * (2**attempt))
    raise last if last else RuntimeError("completion failed")


def gateway_seat_caller(
    client: Any,
    *,
    max_tokens: int = 8,
    temperature: float = 0.0,
    retries: int = 2,
    backoff_base: float = 0.5,
) -> SeatCaller:
    """Build a SeatCaller over an OpenAI-compatible `client`. Retries transient
    failures with backoff; if a seat is still unreachable after retries it
    abstains ('tie') so one flaky seat never crashes a verdict. Run `health_check`
    first to QUARANTINE persistently-down seats (don't let an outage masquerade
    as a tie — AIN-546)."""

    def call(seat: Seat, first: str, second: str) -> str:
        try:
            resp = _complete_with_retry(
                client,
                seat.model_slug,
                build_pairwise_messages(first, second),
                max_tokens=max_tokens,
                temperature=temperature,
                retries=retries,
                backoff_base=backoff_base,
            )
            return parse_pick(resp.choices[0].message.content)
        except Exception:  # noqa: BLE001 — exhausted retries → abstain
            return "tie"

    return call


def health_check(
    client: Any,
    seats: tuple[Seat, ...],
    *,
    retries: int = 1,
    backoff_base: float = 0.5,
) -> tuple[list[Seat], list[Seat]]:
    """Probe each seat once (with retry) → ``(reachable, unreachable)``. The
    caller runs the Council on the reachable set and RECORDS the unreachable ones
    on each verdict so a degraded roster is visible, never silently dropped."""
    ping = [{"role": "user", "content": "Reply one word: ok"}]
    reachable: list[Seat] = []
    unreachable: list[Seat] = []
    for seat in seats:
        try:
            _complete_with_retry(
                client,
                seat.model_slug,
                ping,
                max_tokens=4,
                temperature=0.0,
                retries=retries,
                backoff_base=backoff_base,
            )
            reachable.append(seat)
        except Exception:  # noqa: BLE001
            unreachable.append(seat)
    return reachable, unreachable
