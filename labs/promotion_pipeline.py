"""promotion_pipeline.py — shadow evidence + PARK-and-PROPOSE promotion gate (AIN-542).

Composes the selection-layer evidence into a promotion CASE, then PARKS it. It never
promotes: a passing case is written as a *proposal* for the founder's D4. Canary onward
is founder-only.

The chain (the cron wrapper feeds real data):
    refit_policy (D1 shrinkage + D7 allocation)  →  candidate
    dr_ope.evaluate_policy                        →  doubly-robust value + CI per cell
    replay_gate.decide(dr_ope=...)                →  PROMOTE / HOLD (quantitative)
    gate_promotion(...)                           →  propose_canary | hold  (this module)
    → write labs_ope_runs (shadow evidence) + promotion_proposals (the verdict)

## The anchor hinge (non-negotiable)
A candidate's quality signal (D1) and its DR-OPE score are only trustworthy once the
verifiable anchor is LIT: a real κ ≥ 0.60 on sufficient n, with no global promotion hold.
`anchor_status` derives this from the latest `labs_kappa_history` row. Until it is lit,
`gate_promotion` returns NOT-PROMOTABLE *regardless of every other condition* — so the
gate provably cannot clear on a constant/absent anchor. Shadow evidence still accrues
(the DR-OPE run is written), but no canary proposal is emitted.

Reference: ainfera-vault methodology/promotion-gate.md (PARK-and-PROPOSE + anchor hinge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from labs.dr_ope import OPEResult
from labs.replay_gate import ReplayVerdict

# κ must clear this on more than the n=36 bootstrap before any candidate is promotable.
MIN_ANCHOR_KAPPA = 0.60
MIN_ANCHOR_PAIRS = 50


@dataclass(frozen=True)
class AnchorStatus:
    lit: bool
    kappa: float | None
    n_pairs: int | None
    reason: str


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def anchor_status(
    kappa_row: dict[str, Any] | None,
    *,
    min_kappa: float = MIN_ANCHOR_KAPPA,
    min_pairs: int = MIN_ANCHOR_PAIRS,
) -> AnchorStatus:
    """Is the verifiable anchor lit? Derived from the latest `labs_kappa_history` row.

    Unlit when: no κ history at all, a global promotion_hold is set, κ is null/below the
    floor, or n is below the sufficiency bar. `kappa_row` is the latest row as a dict
    (or None when the table is empty).
    """
    if kappa_row is None:
        return AnchorStatus(False, None, None, "anchor_unlit:no_kappa_history")
    kappa = _as_float(kappa_row.get("kappa_teacher"))
    n = kappa_row.get("n_teacher_pairs")
    if kappa_row.get("promotion_hold"):
        return AnchorStatus(False, kappa, n, "anchor_unlit:global_promotion_hold")
    if kappa is None:
        return AnchorStatus(False, None, n, "anchor_unlit:kappa_null")
    if kappa < min_kappa:
        return AnchorStatus(
            False, kappa, n, f"anchor_unlit:kappa_below_floor({kappa:.3f}<{min_kappa})"
        )
    if n is None or n < min_pairs:
        return AnchorStatus(
            False, kappa, n, f"anchor_unlit:insufficient_n({n}<{min_pairs})"
        )
    return AnchorStatus(True, kappa, n, "lit")


@dataclass(frozen=True)
class PromotionDecision:
    promotable: bool
    action: str  # "propose_canary" | "hold"
    status: str  # "proposed" | "not_promotable"
    rationale: str
    blocking: tuple[str, ...]  # the conditions that failed (empty when promotable)


def gate_promotion(
    *,
    anchor: AnchorStatus,
    replay: ReplayVerdict,
    customer_safety_ok: bool,
    cuped_ready: bool,
    customer_under_baseline: bool,
) -> PromotionDecision:
    """PARK + PROPOSE. A canary proposal is emitted ONLY when every condition holds:
    anchor lit · replay_gate PROMOTE (on the DR-OPE CI) · CS guards · CUPED ready ·
    customer all-in < baseline. Anything missing → NOT-PROMOTABLE (hold).

    The anchor hinge is evaluated first and is non-overridable: an unlit anchor always
    blocks, so the gate cannot clear on a constant/absent anchor.
    """
    blocking: list[str] = []
    if not anchor.lit:
        blocking.append(anchor.reason)
    if replay.decision != "PROMOTE":
        blocking.append(f"replay_gate:{replay.halted_reason or 'hold'}")
    if not customer_safety_ok:
        blocking.append("customer_safety_guards")
    if not cuped_ready:
        blocking.append("cuped_not_ready")
    if not customer_under_baseline:
        blocking.append("customer_all_in_not_below_baseline")

    if not blocking:
        return PromotionDecision(
            promotable=True,
            action="propose_canary",
            status="proposed",
            rationale=(
                "all gates clear: anchor lit (κ="
                f"{anchor.kappa}, n={anchor.n_pairs}), replay_gate PROMOTE on the DR-OPE "
                "CI, CS + CUPED ready, customer all-in < baseline"
            ),
            blocking=(),
        )
    return PromotionDecision(
        promotable=False,
        action="hold",
        status="not_promotable",
        rationale="PARKED — " + "; ".join(blocking),
        blocking=tuple(blocking),
    )


def _confidence(ope: OPEResult) -> float:
    """ESS / n — a 0..1 trust score (low ⇒ degenerate importance weights)."""
    return round(ope.ess / ope.n, 6) if ope.n else 0.0


def ope_run_row(
    *,
    model_slug: str,
    cell: str | None,
    ope: OPEResult,
    decision: PromotionDecision,
    replay: ReplayVerdict,
    anchor: AnchorStatus,
    shadow: bool = True,
) -> dict[str, Any]:
    """A `labs_ope_runs` row — the shadow DR-OPE evidence (written every run, even when
    not promotable). `promote` is the gate's verdict (never auto-acted on)."""
    return {
        "model_slug": model_slug,
        "cell": cell,
        "n": ope.n,
        "value": round(ope.v_dr, 6),
        "ci_low": round(ope.ci_low, 6),
        "ci_high": round(ope.ci_high, 6),
        "confidence": _confidence(ope),
        "shadow": shadow,
        "promote": decision.promotable,
        "gate": {
            "replay_decision": replay.decision,
            "replay_halted_reason": replay.halted_reason,
            "anchor_lit": anchor.lit,
            "anchor_reason": anchor.reason,
            "lift": round(ope.lift, 6),
            "ess": round(ope.ess, 4),
            "blocking": list(decision.blocking),
        },
        "rationale": decision.rationale,
    }


def proposal_row(
    *,
    model_slug: str,
    cell: str | None,
    decision: PromotionDecision,
    ope: OPEResult,
    current_q_prior: float | None,
    proposed_q_prior: float | None,
    observed_mean_reward: float | None,
    observed_samples: int,
    counterfactual_mean_reward: float | None,
    shadow: bool = True,
) -> dict[str, Any]:
    """A `promotion_proposals` row — emitted ONLY for a PROMOTABLE decision (a canary
    proposal for the founder's D4). Maps to the table's vocabulary: `action='promote'`,
    `status='proposed'`. A NOT-PROMOTABLE run writes NO proposal row — its verdict lives
    in `labs_ope_runs` (`promote=false` + the blocking reasons). This is why an unlit
    anchor leaves `promotion_proposals` empty by construction."""
    if not decision.promotable:
        raise ValueError(
            "proposal_row is only for promotable decisions; a parked run records its "
            "verdict in labs_ope_runs, not promotion_proposals"
        )
    return {
        "model_slug": model_slug,
        "cell": cell,
        "action": "promote",
        "current_q_prior": current_q_prior,
        "proposed_q_prior": proposed_q_prior,
        "observed_mean_reward": observed_mean_reward,
        "observed_samples": observed_samples,
        "counterfactual_mean_reward": counterfactual_mean_reward,
        "rationale": decision.rationale,
        "status": "proposed",
        "shadow": shadow,
        "ci_low": round(ope.ci_low, 6),
        "ci_high": round(ope.ci_high, 6),
        "confidence": _confidence(ope),
    }


__all__ = [
    "MIN_ANCHOR_KAPPA",
    "MIN_ANCHOR_PAIRS",
    "AnchorStatus",
    "PromotionDecision",
    "anchor_status",
    "gate_promotion",
    "ope_run_row",
    "proposal_row",
]
