"""AIN-546 · Pointwise cross-family Council for the OPERATIONAL judge_score.

The pairwise Council (``labs.council``) produces Tier-B *soft labels* (A-vs-B,
Dawid–Skene) for the non-verifiable slice. This module is its complement: a
**pointwise** panel that scores a SINGLE response 1–5 — a drop-in replacement for
the single-seat bulk judge that currently writes ``routing_outcomes.judge_score``
(today 100% ``nemotron-3-super``, which fails the teacher gate and was quarantined).

Why pointwise here, when the council is pairwise elsewhere:

* The operational ``judge_score`` IS pointwise 1–5, and the promotion anchor κ
  (``api/scripts/bulk_judge_worker._anchor_agreement``) pairs that score —
  binarized at ``COUNCIL_PASS_MIN_SCORE`` — against the execution-verifiable
  reward. A pointwise panel is therefore a *drop-in*: no change to the κ pairing,
  the binarization, or the schema. A pairwise panel would need a reference output
  per item (which the verify rows don't carry).
* It reuses the cross-family roster wholesale — ``COUNCIL_SEATS`` (5 disjoint
  families), ``eligible_seats`` (self-preference family-exclusion), ``family_of``,
  and ``health_check`` — so a verdict here also satisfies the AIN-546 floor
  (≥3 seats / ≥2 families / populated ``excluded_seats``).

The single-seat judge is what made the lit κ untrustworthy (one rater, no family
diversity, no self-preference firewall). This panel fixes the *rater*; the κ math
is unchanged.

Pure + fully unit-tested. The live model calls are the seam (``PointwiseSeatCaller``
/ :func:`gateway_pointwise_caller`); aggregation has no I/O. Optional per-seat
reliability weighting is a hook for the Step-4 anchor-calibrated reliability — left
uniform by default (a per-item panel can't estimate reliability on its own).
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from labs.council_seats import COUNCIL_SEATS, Seat, eligible_seats

# MUST match api/scripts/bulk_judge_worker.COUNCIL_PASS_MIN_SCORE — this is the
# pass/fail axis the promotion-anchor κ binarizes the written judge_score on. If
# these drift, the panel and the κ gate disagree on what "pass" means.
COUNCIL_PASS_MIN_SCORE = 3

# A seat, shown the TASK and ONE response, returns an integer 1–5 — or ``None`` to
# ABSTAIN (unreachable / unparseable). Abstention is never coerced to a score: a
# confused or down seat must not inject a fabricated label (mirrors the pairwise
# caller's 'tie' abstention, AIN-546).
PointwiseSeatCaller = Callable[[Seat, str, str], "int | None"]

# Neutral default. The real rubric (the moat) is injected via env, exactly as
# labs.eval_harness.judge does — never hard-coded in this public repo.
_DEFAULT_RUBRIC = (
    "Score how well the RESPONSE completes the TASK on a 1-5 integer scale: "
    "1=unusable/wrong, 2=poor, 3=adequate, 4=good, 5=excellent. "
    "Reply with the single integer only."
)


def rubric() -> str:
    return os.environ.get("LABS_COUNCIL_RUBRIC", _DEFAULT_RUBRIC)


def build_pointwise_messages(task: str, response: str) -> list[dict[str, str]]:
    user = (
        f"TASK:\n{task}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "Score the RESPONSE for that task. Reply with the single integer 1-5 only."
    )
    return [
        {"role": "system", "content": rubric()},
        {"role": "user", "content": user},
    ]


_SCORE_RE = re.compile(r"[1-5]")


def parse_score(text: str | None) -> int | None:
    """Map a seat's reply to an int 1–5, or ``None`` if unparseable. ``None`` ⇒
    abstain — the seat is dropped from the panel rather than counted as a guess."""
    if not text:
        return None
    m = _SCORE_RE.search(text.strip())
    return int(m.group(0)) if m else None


# ── verdict ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PointwiseVerdict:
    """One pointwise Council verdict over a single response.

    ``consensus_score`` is the value a worker writes to ``routing_outcomes.judge_score``
    (the anchor κ then binarizes it at ``COUNCIL_PASS_MIN_SCORE``). ``passed`` exposes
    that same binarization here for convenience. ``None`` consensus ⇒ no eligible seat
    produced a score (whole panel abstained / excluded)."""

    item_id: str
    consensus_score: float | None
    passed: bool | None
    confidence: float  # 1 - dispersion; rough certainty of the consensus
    dispersion: float  # normalized score spread in [0,1]; high → Tier C deliberation
    n_seats: int
    n_families: int
    excluded_seats: list[str]
    seat_scores: dict[str, int]
    unreachable_seats: list[str] = field(default_factory=list)

    @property
    def meets_floor(self) -> bool:
        """AIN-546 acceptance: ≥3 seats from ≥2 families scored this verdict."""
        return self.n_seats >= 3 and self.n_families >= 2

    @property
    def degraded(self) -> bool:
        """FLAG (don't certify): floor unmet and/or seats were unreachable. A
        degraded roster must be VISIBLE — an outage must never silently pass as a
        certified verdict (AIN-546)."""
        return (not self.meets_floor) or bool(self.unreachable_seats)


def _dispersion(scores: list[int]) -> float:
    """Normalized spread of 1–5 scores → [0,1]. 0 = unanimous; ~1 = maximally
    split ({1}∪{5}). Population stdev normalized by 2.0 (the stdev of a 50/50
    {1,5} split), clamped."""
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    return min(1.0, math.sqrt(var) / 2.0)


def score_response(
    item_id: str,
    task: str,
    response: str,
    candidate_family: str,
    seat_caller: PointwiseSeatCaller,
    *,
    seats: tuple[Seat, ...] = COUNCIL_SEATS,
    pass_min: int = COUNCIL_PASS_MIN_SCORE,
    seat_reliability: dict[str, float] | None = None,
) -> PointwiseVerdict:
    """Run the pointwise panel on one response.

    Self-preference firewall: every seat whose family == ``candidate_family`` is
    excluded (a seat never scores its own family's output). Each eligible seat is
    asked for a 1–5; abstentions (``None``) are recorded as unreachable, never
    scored. The consensus is the (optionally reliability-weighted) mean; ``passed``
    binarizes it at ``pass_min`` — the same axis the promotion-anchor κ uses."""
    eligible, excluded = eligible_seats({candidate_family}, seats)
    seat_scores: dict[str, int] = {}
    unreachable: list[str] = []
    families: set[str] = set()
    for seat in eligible:
        score = seat_caller(seat, task, response)
        if score is None:
            unreachable.append(seat.persona)
            continue
        seat_scores[seat.persona] = int(score)
        families.add(seat.family)

    if not seat_scores:
        return PointwiseVerdict(
            item_id, None, None, 0.0, 1.0, 0, 0,
            [s.persona for s in excluded], {}, unreachable,
        )

    scores = list(seat_scores.values())
    if seat_reliability:
        # reliability-weighted mean (Step-4 hook); unknown seats default to 1.0.
        wsum = sum(seat_reliability.get(p, 1.0) for p in seat_scores)
        consensus = (
            sum(seat_reliability.get(p, 1.0) * s for p, s in seat_scores.items()) / wsum
            if wsum > 0
            else sum(scores) / len(scores)
        )
    else:
        consensus = sum(scores) / len(scores)

    dispersion = _dispersion(scores)
    return PointwiseVerdict(
        item_id=item_id,
        consensus_score=round(consensus, 4),
        passed=consensus >= pass_min,
        confidence=round(1.0 - dispersion, 4),
        dispersion=round(dispersion, 4),
        n_seats=len(seat_scores),
        n_families=len(families),
        excluded_seats=[s.persona for s in excluded],
        seat_scores=seat_scores,
        unreachable_seats=unreachable,
    )


def needs_deliberation(verdict: PointwiseVerdict, threshold: float = 0.6) -> bool:
    """Tier C trigger: high inter-seat disagreement → debate / down-weight rather
    than certify the consensus."""
    return verdict.dispersion >= threshold


def run_pointwise_council(
    items: list[tuple[str, str, str, str]],
    seat_caller: PointwiseSeatCaller,
    *,
    seats: tuple[Seat, ...] = COUNCIL_SEATS,
    pass_min: int = COUNCIL_PASS_MIN_SCORE,
    unreachable_seats: tuple[str, ...] = (),
    seat_reliability: dict[str, float] | None = None,
) -> list[PointwiseVerdict]:
    """Score a batch of ``(item_id, task, response, candidate_family)`` rows.

    ``unreachable_seats`` (from a prior :func:`labs.seat_caller.health_check`) is
    folded into every verdict's roster so a degraded panel is visible batch-wide,
    not silently dropped."""
    out: list[PointwiseVerdict] = []
    for item_id, task, response, candidate_family in items:
        v = score_response(
            item_id, task, response, candidate_family, seat_caller,
            seats=seats, pass_min=pass_min, seat_reliability=seat_reliability,
        )
        if unreachable_seats:
            merged = sorted(set(v.unreachable_seats) | set(unreachable_seats))
            v = PointwiseVerdict(
                v.item_id, v.consensus_score, v.passed, v.confidence, v.dispersion,
                v.n_seats, v.n_families, v.excluded_seats, v.seat_scores, merged,
            )
        out.append(v)
    return out


# ── live seam (OpenAI-compatible; injected so aggregation stays testable) ─────


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
    """One completion with exponential-backoff retry on transient errors (the
    gateway 502s seen live). Raises the last error once retries are exhausted —
    the caller maps that to abstention."""
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


def gateway_pointwise_caller(
    client: Any,
    *,
    max_tokens: int = 8,
    temperature: float = 0.0,
    retries: int = 2,
    backoff_base: float = 0.5,
) -> PointwiseSeatCaller:
    """Build a :data:`PointwiseSeatCaller` over an OpenAI-compatible ``client``.
    A seat still unreachable after retries ABSTAINS (returns ``None``) so one flaky
    seat never crashes — or silently biases — a verdict. Run
    :func:`labs.seat_caller.health_check` first to quarantine persistently-down seats."""

    def call(seat: Seat, task: str, response: str) -> int | None:
        try:
            resp = _complete_with_retry(
                client,
                seat.model_slug,
                build_pointwise_messages(task, response),
                max_tokens=max_tokens,
                temperature=temperature,
                retries=retries,
                backoff_base=backoff_base,
            )
            return parse_score(resp.choices[0].message.content)
        except Exception:  # noqa: BLE001 — exhausted retries → abstain
            return None

    return call
