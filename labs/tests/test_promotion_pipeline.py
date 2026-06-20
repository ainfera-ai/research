"""AIN-542 — promotion gate: anchor hinge + PARK-and-PROPOSE."""

from __future__ import annotations

from labs.dr_ope import OPEResult
from labs.promotion_pipeline import (
    anchor_status,
    gate_promotion,
    ope_run_row,
    proposal_row,
)
from labs.replay_gate import ReplayVerdict


def _ope(ci_low: float = 0.02, n: int = 200, ess: float = 150.0) -> OPEResult:
    return OPEResult(
        v_dr=0.62,
        v_logged=0.6,
        lift=0.02,
        ci_low=ci_low,
        ci_high=ci_low + 0.04,
        n=n,
        ess=ess,
        mean_weight=1.0,
    )


def _replay(decision: str = "PROMOTE", halted: str | None = None) -> ReplayVerdict:
    return ReplayVerdict(
        decision=decision,
        incumbent_version="v1",
        candidate_version="v2",
        overall_delta_pct=1.0,
        guard_delta_met=True,
        guard_no_regression=True,
        guard_exploration_floor=True,
        guard_min_sample=True,
        halted_reason=halted,
    )


def _lit():
    return anchor_status(
        {"kappa_teacher": 0.72, "n_teacher_pairs": 120, "promotion_hold": False}
    )


# ── anchor_status ──────────────────────────────────────────────────────────


def test_anchor_unlit_when_no_history() -> None:
    a = anchor_status(None)
    assert not a.lit and "no_kappa_history" in a.reason


def test_anchor_unlit_on_global_hold() -> None:
    a = anchor_status(
        {"kappa_teacher": 0.9, "n_teacher_pairs": 200, "promotion_hold": True}
    )
    assert not a.lit and "global_promotion_hold" in a.reason


def test_anchor_unlit_below_kappa_floor() -> None:
    a = anchor_status(
        {"kappa_teacher": 0.4, "n_teacher_pairs": 200, "promotion_hold": False}
    )
    assert not a.lit and "kappa_below_floor" in a.reason


def test_anchor_unlit_insufficient_n() -> None:
    a = anchor_status(
        {"kappa_teacher": 0.8, "n_teacher_pairs": 36, "promotion_hold": False}
    )
    assert not a.lit and "insufficient_n" in a.reason


def test_anchor_lit() -> None:
    a = _lit()
    assert a.lit and a.kappa == 0.72 and a.n_pairs == 120


# ── the hinge ──────────────────────────────────────────────────────────────


def test_unlit_anchor_blocks_even_with_all_else_green() -> None:
    # every other condition passes, but the anchor is unlit → still NOT-PROMOTABLE.
    d = gate_promotion(
        anchor=anchor_status(None),
        replay=_replay("PROMOTE"),
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    assert d.promotable is False
    assert d.action == "hold" and d.status == "not_promotable"
    assert any("anchor_unlit" in b for b in d.blocking)


def test_all_green_proposes_canary() -> None:
    d = gate_promotion(
        anchor=_lit(),
        replay=_replay("PROMOTE"),
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    assert d.promotable is True
    assert d.action == "propose_canary" and d.status == "proposed"


def test_replay_hold_blocks() -> None:
    d = gate_promotion(
        anchor=_lit(),
        replay=_replay("HOLD", "delta_below_floor"),
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    assert not d.promotable and any("replay_gate" in b for b in d.blocking)


def test_customer_over_baseline_blocks() -> None:
    d = gate_promotion(
        anchor=_lit(),
        replay=_replay("PROMOTE"),
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=False,
    )
    assert not d.promotable and "customer_all_in_not_below_baseline" in d.blocking


# ── row builders ───────────────────────────────────────────────────────────


def test_ope_run_row_shape() -> None:
    a = anchor_status(None)
    r = _replay("HOLD", "x")
    d = gate_promotion(
        anchor=a,
        replay=r,
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    row = ope_run_row(
        model_slug="m", cell="code:cost", ope=_ope(), decision=d, replay=r, anchor=a
    )
    assert row["promote"] is False and row["shadow"] is True
    assert (
        row["n"] == 200
        and "blocking" in row["gate"]
        and row["gate"]["anchor_lit"] is False
    )


def test_proposal_row_shape() -> None:
    a = _lit()
    r = _replay("PROMOTE")
    d = gate_promotion(
        anchor=a,
        replay=r,
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    row = proposal_row(
        model_slug="m",
        cell="code:cost",
        decision=d,
        ope=_ope(),
        current_q_prior=0.8,
        proposed_q_prior=0.85,
        observed_mean_reward=0.7,
        observed_samples=200,
        counterfactual_mean_reward=0.72,
    )
    # maps to the table vocabulary (action ∈ {promote,demote,list}; status ∈ {proposed,...})
    assert (
        row["status"] == "proposed"
        and row["action"] == "promote"
        and row["shadow"] is True
    )


def test_proposal_row_rejects_not_promotable() -> None:
    # a parked (unlit-anchor) decision must NOT produce a proposal row — it records its
    # verdict in labs_ope_runs instead, so promotion_proposals stays empty by construction.
    import pytest

    a = anchor_status(None)
    r = _replay("PROMOTE")
    d = gate_promotion(
        anchor=a,
        replay=r,
        customer_safety_ok=True,
        cuped_ready=True,
        customer_under_baseline=True,
    )
    with pytest.raises(ValueError):
        proposal_row(
            model_slug="m",
            cell="code:cost",
            decision=d,
            ope=_ope(),
            current_q_prior=0.8,
            proposed_q_prior=0.85,
            observed_mean_reward=0.7,
            observed_samples=200,
            counterfactual_mean_reward=0.72,
        )
