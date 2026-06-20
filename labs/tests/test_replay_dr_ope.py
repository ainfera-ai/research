"""replay_gate × DR-OPE — the optional quantitative 5th guard (AIN-542)."""

from __future__ import annotations

from labs.dr_ope import OPEResult
from labs.replay_gate import decide


def _cell(
    task_type: str, candidate: str, dnc: float, n: int = 100, explore: float = 0.10
):
    return {
        "task_type": task_type,
        "candidate": candidate,
        "done_and_cheaper_pct": dnc,
        "n_held_out": n,
        "explore_pct": explore,
    }


# 4 frozen guards all pass: +1.0pp, ample n, explore ok, no regression.
INC = [_cell("code", "opus", 60.0)]
CAND = [_cell("code", "opus", 61.0)]


def _ope(*, ci_low: float, ess: float, n: int = 100) -> OPEResult:
    return OPEResult(
        v_dr=0.6 + ci_low,
        v_logged=0.6,
        lift=ci_low + 0.02,
        ci_low=ci_low,
        ci_high=ci_low + 0.05,
        n=n,
        ess=ess,
        mean_weight=1.0,
    )


def _decide(dr_ope=None):
    return decide(
        incumbent_cells=INC,
        candidate_cells=CAND,
        incumbent_version="v1",
        candidate_version="v2",
        dr_ope=dr_ope,
    )


def test_no_dr_ope_is_byte_identical_v0() -> None:
    v = _decide()
    assert v.decision == "PROMOTE"
    assert v.guard_dr_ope is None
    assert "dr_ope" not in v.to_json()  # v0 verdict schema untouched


def test_positive_lift_promotes() -> None:
    v = _decide(_ope(ci_low=0.01, ess=80))  # CI lower bound > 0, healthy ESS
    assert v.decision == "PROMOTE"
    assert v.guard_dr_ope is True
    assert '"dr_ope_positive": true' in v.to_json()


def test_lift_ci_not_positive_holds() -> None:
    v = _decide(_ope(ci_low=-0.01, ess=80))  # CI includes 0 → not confidently better
    assert v.decision == "HOLD"
    assert v.guard_dr_ope is False
    assert v.halted_reason.startswith("dr_ope_lift_ci_not_positive")


def test_degenerate_ess_holds() -> None:
    v = _decide(_ope(ci_low=0.02, ess=5))  # ess 5 < 10%·100 → degenerate weights
    assert v.decision == "HOLD"
    assert v.halted_reason.startswith("dr_ope_ess_degenerate")


def test_frozen_guard_failure_takes_precedence() -> None:
    # candidate only +0.3pp (< 0.5 frozen floor): the frozen guard holds, NOT the DR-OPE one,
    # even though DR-OPE would pass — DR-OPE can only ADD strictness, never override.
    v = decide(
        incumbent_cells=INC,
        candidate_cells=[_cell("code", "opus", 60.3)],
        incumbent_version="v1",
        candidate_version="v2",
        dr_ope=_ope(ci_low=0.05, ess=90),
    )
    assert v.decision == "HOLD"
    assert v.halted_reason.startswith("delta_below_floor")
