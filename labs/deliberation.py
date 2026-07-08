"""AIN-542 Step 5 · Tier C — dispersion → deliberation pipeline.

When the Council's inter-seat disagreement is high (dispersion above a
threshold), the verdict is **low-confidence**.  The constitution says:

    High disagreement = low confidence. Escalate or down-weight.

This module implements that contract.  It is agnostic to the verdict type:
both :class:`labs.council.CouncilVerdict` (pairwise, entropy-dispersion) and
:class:`labs.council_pointwise.PointwiseVerdict` (pointwise, stdev-dispersion)
expose the four fields the pipeline needs — ``item_id``, ``dispersion``,
``confidence``, and ``label``/``passed`` — and the pipeline keys off those
fields plus a small adapter, so it never imports a concrete verdict class.

Two actions, never both silently:

1. **Down-weight** — multiply the reward / confidence by a factor < 1 so the
   noisy verdict carries less training signal.  The multiplier is a smooth
   function of how far above threshold the dispersion is.

2. **Escalate** — send the item to a *second-round* judgment with a different
   or expanded seat roster.  The round-2 verdict replaces round-1 if it is
   itself low-dispersion; if round-2 *also* disagrees, the item is
   permanently down-weighted and marked ``escalation_exhausted``.

Escalation history is tracked per item (``EscalationRecord`` +
``DeliberationOutcome``) so callers can audit why a verdict was down-weighted
and never re-escalate an already-exhausted item.

Pure + fully unit-tested.  The live second-round Council call is the seam
(``RejudgeCallback``); the pipeline has no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ── config ───────────────────────────────────────────────────────────────────

DEFAULT_DISPERSION_THRESHOLD = 0.4
"""Items with dispersion > this are flagged low-confidence and enter the
pipeline.  The pairwise council uses entropy-normalized dispersion in [0,1]
(0.85 is a very high bar); the pointwise council uses stdev-normalized spread
(0.6 is its default).  0.4 is a conservative single default — callers should
pass the verdict-specific threshold explicitly when they know the source."""

DEFAULT_DOWNWEIGHT_FLOOR = 0.2
"""The down-weight multiplier never goes below this, so a noisy verdict is
reduced but never zeroed (a zero-confidence item would be indistinguishable
from a missing one in downstream training)."""

MAX_ESCALATION_ROUNDS = 2
"""Hard cap on second-round panels.  Two rounds of disagreement = the Council
genuinely cannot decide → permanent down-weight, not infinite retry."""


# ── verdict adapter (structural typing) ──────────────────────────────────────

@runtime_checkable
class _DispersionVerdict(Protocol):
    """Structural interface any verdict must satisfy to enter the pipeline.

    Both ``CouncilVerdict`` and ``PointwiseVerdict`` conform — the pipeline
    never imports them, so it works with either and with any future verdict
    type that exposes these four fields.
    """

    item_id: str
    dispersion: float
    confidence: float


@dataclass(frozen=True)
class VerdictView:
    """Minimal, frozen projection of a verdict for the pipeline.

    Extracting the relevant fields up front means the pipeline never holds a
    reference to the original (possibly mutable or heavyweight) verdict object
    — it deliberates on a small snapshot.
    """

    item_id: str
    dispersion: float
    confidence: float
    label: Any  # Vote | bool | None — whatever the verdict considers its decision
    n_seats: int
    n_families: int

    @classmethod
    def from_verdict(cls, v: Any) -> "VerdictView":
        """Adapt any object with the four required fields."""
        label = getattr(v, "label", None)
        if label is None:
            label = getattr(v, "passed", None)
        return cls(
            item_id=v.item_id,
            dispersion=float(v.dispersion),
            confidence=float(v.confidence),
            label=label,
            n_seats=int(getattr(v, "n_seats", 0)),
            n_families=int(getattr(v, "n_families", 0)),
        )


# ── down-weighting ──────────────────────────────────────────────────────────

def down_weight_multiplier(
    dispersion: float,
    threshold: float = DEFAULT_DISPERSION_THRESHOLD,
    floor: float = DEFAULT_DOWNWEIGHT_FLOOR,
) -> float:
    """Smooth down-weight factor in ``(floor, 1.0]``.

    At ``dispersion == threshold`` the multiplier is 1.0 (no penalty — the
    item just barely entered the pipeline).  As dispersion → 1.0 the
    multiplier → ``floor``.  The curve is linear in the excess:

        multiplier = 1 - (1 - floor) * (dispersion - threshold) / (1 - threshold)

    Items *below* threshold get 1.0 (no penalty).
    """
    if dispersion <= threshold:
        return 1.0
    excess = (dispersion - threshold) / max(1.0 - threshold, 1e-9)
    return max(floor, 1.0 - (1.0 - floor) * excess)


# ── escalation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EscalationRecord:
    """One round of escalation for one item."""

    round_number: int  # 1 = first escalation (second-round panel)
    roster_label: str  # human description of the round-2 panel
    dispersion: float
    confidence: float
    label: Any
    still_low_confidence: bool


@dataclass
class DeliberationOutcome:
    """Final result of the pipeline for one item.

    ``final_confidence`` is the value downstream training should use — it is
    the round-2 confidence (if escalation happened) multiplied by the
    down-weight factor.  ``history`` is the full escalation trace for audit.
    """

    item_id: str
    original_dispersion: float
    original_confidence: float
    low_confidence: bool
    action: str  # 'certify' | 'down_weight' | 'escalate' | 'escalation_exhausted'
    down_weight: float  # multiplier applied (1.0 if none)
    final_confidence: float
    final_label: Any
    history: list[EscalationRecord] = field(default_factory=list)

    @property
    def was_escalated(self) -> bool:
        return bool(self.history)


# ── re-judge seam ────────────────────────────────────────────────────────────

RejudgeCallback = Callable[[str], Any]
"""Given an ``item_id``, produce a *new* verdict (round-2 panel).  The
callback is responsible for selecting a different/expanded roster and running
the council — the pipeline only decides *whether* to call it and records the
result.  Returns any object conforming to :class:`_DispersionVerdict`."""


# ── pipeline ─────────────────────────────────────────────────────────────────

def is_low_confidence(
    dispersion: float,
    threshold: float = DEFAULT_DISPERSION_THRESHOLD,
) -> bool:
    """Tier C trigger: dispersion above threshold → low confidence."""
    return dispersion > threshold


def deliberate(
    verdict: Any,
    *,
    threshold: float = DEFAULT_DISPERSION_THRESHOLD,
    down_weight_floor: float = DEFAULT_DOWNWEIGHT_FLOOR,
    rejudge: RejudgeCallback | None = None,
    max_rounds: int = MAX_ESCALATION_ROUNDS,
    _history: list[EscalationRecord] | None = None,
) -> DeliberationOutcome:
    """Run the dispersion → deliberation pipeline on one verdict.

    Decision tree:

    * **dispersion ≤ threshold** → ``certify``.  Confidence unchanged.
    * **dispersion > threshold, no ``rejudge``** → ``down_weight``.  Confidence
      multiplied by :func:`down_weight_multiplier`.
    * **dispersion > threshold, ``rejudge`` provided** → ``escalate``.  The
      re-judge callback is called; if the round-2 verdict is still
      low-confidence, repeat up to ``max_rounds``.  If every round is
      low-confidence → ``escalation_exhausted`` (permanent down-weight on the
      *last* round's confidence).  If any round produces a high-confidence
      verdict, that round's verdict is certified (with any residual
      down-weight from remaining dispersion).

    The outcome's ``final_confidence`` is always the value to feed downstream.
    """
    view = VerdictView.from_verdict(verdict)
    _history = _history if _history is not None else []

    # ── base case: dispersion within tolerance → certify ─────────────────
    if not is_low_confidence(view.dispersion, threshold):
        return DeliberationOutcome(
            item_id=view.item_id,
            original_dispersion=view.dispersion,
            original_confidence=view.confidence,
            low_confidence=False,
            action="certify",
            down_weight=1.0,
            final_confidence=view.confidence,
            final_label=view.label,
            history=_history,
        )

    # ── low confidence: escalate if we have a rejudge callback and budget ─
    round_num = len(_history) + 1

    if rejudge is not None and round_num <= max_rounds:
        round2_verdict = rejudge(view.item_id)
        r2 = VerdictView.from_verdict(round2_verdict)
        still_low = is_low_confidence(r2.dispersion, threshold)

        roster_label = getattr(round2_verdict, "roster_label", f"round-{round_num + 1}")
        _history.append(
            EscalationRecord(
                round_number=round_num,
                roster_label=roster_label,
                dispersion=r2.dispersion,
                confidence=r2.confidence,
                label=r2.label,
                still_low_confidence=still_low,
            )
        )

        if not still_low:
            # round-2 resolved it → certify the round-2 verdict
            return DeliberationOutcome(
                item_id=view.item_id,
                original_dispersion=view.dispersion,
                original_confidence=view.confidence,
                low_confidence=True,
                action="escalate",
                down_weight=1.0,
                final_confidence=r2.confidence,
                final_label=r2.label,
                history=_history,
            )

        # round-2 still low → recurse (try round 3, or exhaust)
        if round_num < max_rounds:
            return deliberate(
                round2_verdict,
                threshold=threshold,
                down_weight_floor=down_weight_floor,
                rejudge=rejudge,
                max_rounds=max_rounds,
                _history=_history,
            )

        # budget exhausted → permanent down-weight on the last round's verdict
        dw = down_weight_multiplier(r2.dispersion, threshold, down_weight_floor)
        return DeliberationOutcome(
            item_id=view.item_id,
            original_dispersion=view.dispersion,
            original_confidence=view.confidence,
            low_confidence=True,
            action="escalation_exhausted",
            down_weight=round(dw, 4),
            final_confidence=round(r2.confidence * dw, 4),
            final_label=r2.label,
            history=_history,
        )

    # ── low confidence, no escalation available → down-weight ─────────────
    dw = down_weight_multiplier(view.dispersion, threshold, down_weight_floor)
    return DeliberationOutcome(
        item_id=view.item_id,
        original_dispersion=view.dispersion,
        original_confidence=view.confidence,
        low_confidence=True,
        action="down_weight",
        down_weight=round(dw, 4),
        final_confidence=round(view.confidence * dw, 4),
        final_label=view.label,
        history=_history,
    )


# ── batch helper ────────────────────────────────────────────────────────────

def deliberate_batch(
    verdicts: list[Any],
    *,
    threshold: float = DEFAULT_DISPERSION_THRESHOLD,
    down_weight_floor: float = DEFAULT_DOWNWEIGHT_FLOOR,
    rejudge: RejudgeCallback | None = None,
    max_rounds: int = MAX_ESCALATION_ROUNDS,
) -> list[DeliberationOutcome]:
    """Run the pipeline over a batch of verdicts.

    Each item is deliberated independently.  Items already in the history of
    a prior call are *not* re-deliberated here — the caller controls batch
    identity.  The rejudge callback, if provided, is shared across all items
    (it decides the round-2 roster per item internally).
    """
    return [
        deliberate(
            v,
            threshold=threshold,
            down_weight_floor=down_weight_floor,
            rejudge=rejudge,
            max_rounds=max_rounds,
        )
        for v in verdicts
    ]


# ── escalation roster selection ─────────────────────────────────────────────

def build_escalation_roster(
    original_seats: tuple[Any, ...],
    spark_pool: tuple[Any, ...],
    *,
    expand: bool = True,
) -> tuple[Any, ...]:
    """Select a different/expanded roster for a second-round panel.

    Strategy: if ``expand`` is True, add all spark-pool seats not already in
    the original roster (larger panel → more diversity → potential
    disagreement resolution).  If ``expand`` is False, swap *out* the original
    seats and use *only* the spark pool (different panel entirely — useful
    when the original panel's reliability is suspect).

    The caller (rejudge callback) decides which strategy to use; this helper
    just produces the roster.
    """
    if expand:
        existing_slugs = {s.model_slug for s in original_seats}
        additions = [s for s in spark_pool if s.model_slug not in existing_slugs]
        return tuple(original_seats) + tuple(additions)
    return tuple(spark_pool)
