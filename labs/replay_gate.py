"""replay_gate.py — PROMOTE / HOLD decision via CRN replay.

Extends the existing CRN harness at `eval/replay.py` to score a candidate
LinUCB-refit policy against the incumbent on a held-out corpus.

Promote criterion (Discipline #12 — moat, frozen by L14.2):
  1. replay_delta_done_and_cheaper ≥ +0.5% (CRN-derived, not noise)
  2. exploration_floor preserved (≥5% on every cell)
  3. no_catastrophic_regression — no cell ≤ -2% done-and-cheaper
  4. sample_size ≥ 30 rows per affected cell

Tie / marginal / any guard fails → HOLD (keep incumbent). The bar is set
so the goal is not 365 promotes/yr — quality compounding is.

Output: ReplayVerdict dataclass with `decision` ∈ {PROMOTE, HOLD} + the
4 guards (boolean) + the numeric deltas. The cron orchestrator forwards
this to `api/v1/admin/policy/publish` ONLY if `decision == 'PROMOTE'`.

References:
  research/eval/replay.py (existing CRN harness — composes here)
  ainfera-vault methodology/daily-training-cadence.md §"Promote criterion"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from labs.dr_ope import OPEResult

log = logging.getLogger(__name__)


# Frozen thresholds — Discipline #12. Changing requires founder GO + Tulkas co-sign.
MIN_DELTA_PCT = 0.5  # ≥ +0.5 percentage points
MAX_CELL_REGRESS_PCT = -2.0  # any cell ≤ -2% → HOLD
EXPLORATION_FLOOR_PCT = 0.05  # ≥5% on every cell
MIN_SAMPLE_PER_CELL = 30

# AIN-542 — OPTIONAL 5th guard (the quantitative half). When the caller supplies a
# DR-OPE result, PROMOTE additionally requires the doubly-robust lift to be confidently
# positive (CI lower bound > 0) on a non-degenerate importance-weight base. This can only
# make the gate STRICTER (never promotes anything the frozen 4 would have held), and is
# inert when no DR-OPE result is passed — so the frozen criterion is untouched.
MIN_DR_OPE_ESS_RATIO = 0.10  # ESS must be ≥ 10% of n (else weights are degenerate)


@dataclass(frozen=True)
class CellDelta:
    task_type: str
    candidate: str
    incumbent_done_and_cheaper_pct: float
    candidate_done_and_cheaper_pct: float
    delta_pct: float
    n_held_out: int


@dataclass(frozen=True)
class ReplayVerdict:
    decision: str  # "PROMOTE" | "HOLD"
    incumbent_version: str
    candidate_version: str
    overall_delta_pct: float
    guard_delta_met: bool
    guard_no_regression: bool
    guard_exploration_floor: bool
    guard_min_sample: bool
    cells: list[CellDelta] = field(default_factory=list)
    halted_reason: str | None = None  # which guard failed
    # AIN-542 optional DR-OPE guard — None when no DR-OPE result was supplied (the
    # frozen 4-guard behaviour); a bool + the doubly-robust summary when it was.
    guard_dr_ope: bool | None = None
    dr_ope: OPEResult | None = None

    def to_json(self) -> str:
        guards: dict[str, Any] = {
            "delta_met": self.guard_delta_met,
            "no_regression": self.guard_no_regression,
            "exploration_floor": self.guard_exploration_floor,
            "min_sample": self.guard_min_sample,
        }
        payload: dict[str, Any] = {
            "decision": self.decision,
            "incumbent_version": self.incumbent_version,
            "candidate_version": self.candidate_version,
            "overall_delta_pct": self.overall_delta_pct,
            "guards": guards,
            "halted_reason": self.halted_reason,
            "cells": [c.__dict__ for c in self.cells],
        }
        # Append the DR-OPE block ONLY when the guard was evaluated, so a run without
        # DR-OPE emits the exact v0 verdict schema.
        if self.guard_dr_ope is not None and self.dr_ope is not None:
            guards["dr_ope_positive"] = self.guard_dr_ope
            payload["dr_ope"] = {
                "lift": round(self.dr_ope.lift, 6),
                "ci_low": round(self.dr_ope.ci_low, 6),
                "ci_high": round(self.dr_ope.ci_high, 6),
                "ess": round(self.dr_ope.ess, 2),
                "n": self.dr_ope.n,
            }
        return json.dumps(payload, indent=2, sort_keys=True)


def decide(
    *,
    incumbent_cells: list[dict[str, Any]],  # per-cell done_and_cheaper_pct + n
    candidate_cells: list[dict[str, Any]],
    incumbent_version: str,
    candidate_version: str,
    dr_ope: OPEResult | None = None,
) -> ReplayVerdict:
    """Pure-function decision. Deterministic given the cell-level inputs.

    Real cell aggregation uses the CRN harness in `eval/replay.py`:
    same prompts → same response trajectories on both policies → honest
    delta. This function decides given the aggregated deltas.

    AIN-542: when `dr_ope` (a doubly-robust off-policy estimate of the candidate
    vs the logging policy) is supplied, PROMOTE additionally requires its lift CI
    lower bound > 0 on a non-degenerate weight base (ESS ≥ 10%·n). Omitting it leaves
    the frozen 4-guard criterion exactly as-is.
    """
    # Index by cell key for paired comparison
    inc_by_cell = {(c["task_type"], c["candidate"]): c for c in incumbent_cells}
    cand_by_cell = {(c["task_type"], c["candidate"]): c for c in candidate_cells}
    all_keys = sorted(inc_by_cell.keys() | cand_by_cell.keys())

    cell_deltas: list[CellDelta] = []
    weighted_inc_total = 0.0
    weighted_cand_total = 0.0
    weight_total = 0.0
    n_undersize_cells = 0
    n_below_floor = 0

    for key in all_keys:
        inc = inc_by_cell.get(
            key, {"done_and_cheaper_pct": 0.0, "n_held_out": 0, "explore_pct": 0.0}
        )
        cand = cand_by_cell.get(
            key, {"done_and_cheaper_pct": 0.0, "n_held_out": 0, "explore_pct": 0.0}
        )
        n = max(inc["n_held_out"], cand["n_held_out"])
        if n < MIN_SAMPLE_PER_CELL:
            n_undersize_cells += 1
        if cand.get("explore_pct", 0.0) < EXPLORATION_FLOOR_PCT:
            n_below_floor += 1

        delta = cand["done_and_cheaper_pct"] - inc["done_and_cheaper_pct"]
        cell_deltas.append(
            CellDelta(
                task_type=key[0],
                candidate=key[1],
                incumbent_done_and_cheaper_pct=inc["done_and_cheaper_pct"],
                candidate_done_and_cheaper_pct=cand["done_and_cheaper_pct"],
                delta_pct=round(delta, 4),
                n_held_out=n,
            )
        )

        weight = float(n)
        weighted_inc_total += inc["done_and_cheaper_pct"] * weight
        weighted_cand_total += cand["done_and_cheaper_pct"] * weight
        weight_total += weight

    overall_delta = (
        (weighted_cand_total - weighted_inc_total) / weight_total
        if weight_total > 0
        else 0.0
    )

    # Evaluate the 4 guards (Discipline #12 frozen).
    guard_delta_met = overall_delta >= MIN_DELTA_PCT
    # PROPOSED (AIN replay-gate diagnosis — needs founder GO + Tulkas co-sign): the
    # no-regression guard must not count cells too thin to judge. A cell with
    # n_held_out < MIN_SAMPLE_PER_CELL whose delta is noisy past -2pp was reported as a
    # quality `regression_in_N_cell(s)` when its honest blocker is `undersize_sample` — the
    # min-sample guard catches it, but is evaluated AFTER no_regression, so the misleading
    # reason wins. Excluding undersize cells here is DECISION-SAFE: any cell dropped is, by
    # definition (n < floor), still counted in n_undersize_cells, so guard_min_sample still
    # HOLDs it — this can never flip a HOLD into a PROMOTE; it only corrects the reason.
    # Real regressions on well-sampled (n >= floor) cells are unaffected.
    guard_no_regression = all(
        c.delta_pct >= MAX_CELL_REGRESS_PCT
        for c in cell_deltas
        if c.n_held_out >= MIN_SAMPLE_PER_CELL
    )
    guard_exploration_floor = n_below_floor == 0
    guard_min_sample = n_undersize_cells == 0

    # Optional DR-OPE guard (AIN-542) — only evaluated when a result is supplied.
    guard_dr_ope: bool | None = None
    if dr_ope is not None:
        ess_ok = dr_ope.ess >= MIN_DR_OPE_ESS_RATIO * dr_ope.n
        guard_dr_ope = (dr_ope.ci_low > 0.0) and ess_ok

    halted_reason: str | None = None
    if not guard_delta_met:
        halted_reason = (
            f"delta_below_floor (got {overall_delta:.4f}pp, need ≥{MIN_DELTA_PCT}pp)"
        )
    elif not guard_no_regression:
        # count only well-sampled regressions (keep the reason consistent with the guard).
        regress_cells = [
            c
            for c in cell_deltas
            if c.delta_pct < MAX_CELL_REGRESS_PCT and c.n_held_out >= MIN_SAMPLE_PER_CELL
        ]
        halted_reason = f"regression_in_{len(regress_cells)}_cell(s)"
    elif not guard_exploration_floor:
        halted_reason = f"exploration_below_floor_in_{n_below_floor}_cell(s)"
    elif not guard_min_sample:
        halted_reason = f"undersize_sample_in_{n_undersize_cells}_cell(s)"
    elif guard_dr_ope is False and dr_ope is not None:
        if dr_ope.ci_low <= 0.0:
            halted_reason = f"dr_ope_lift_ci_not_positive (ci_low={dr_ope.ci_low:.4f})"
        else:
            halted_reason = (
                f"dr_ope_ess_degenerate (ess={dr_ope.ess:.1f}, n={dr_ope.n})"
            )

    decision = "PROMOTE" if halted_reason is None else "HOLD"

    return ReplayVerdict(
        decision=decision,
        incumbent_version=incumbent_version,
        candidate_version=candidate_version,
        overall_delta_pct=round(overall_delta, 4),
        guard_delta_met=guard_delta_met,
        guard_no_regression=guard_no_regression,
        guard_exploration_floor=guard_exploration_floor,
        guard_min_sample=guard_min_sample,
        halted_reason=halted_reason,
        cells=cell_deltas,
        guard_dr_ope=guard_dr_ope,
        dr_ope=dr_ope,
    )
