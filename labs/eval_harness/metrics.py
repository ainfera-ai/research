"""metrics.py — the eval metrics + the pre-registered win check.

Pure functions over a list of ArmCall records. Stage 1 ships the math + the
decision contract; Stage 2 feeds it live measurements. No external deps beyond
the stdlib (numpy is available but the Wilson interval is pure arithmetic, so we
keep this module dependency-free and trivially testable).

Metrics (per arm): success rate, cost_per_success, cost_per_call, floor
breaches, n, 95% CI (Wilson score interval on the success proportion).

Win check (arm C, pre-registered — eval spec
decisions/2026-06-15-eval-spec-3arm-cost-outcome.md):
    C.success >= floor (per type)        AND
    C.success >= A.success - EPSILON_PT  AND
    C.cost_per_success < A.cost_per_success AND
    C.cost_per_success < D.cost_per_success
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from labs.eval_harness import config
from labs.eval_harness.arms import ArmCall


# --- confidence interval ---------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at z=1.96).

    Pure arithmetic — no scipy. Returns (low, high) clamped to [0, 1]. n=0 → the
    whole interval (0, 1).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# --- per-arm metrics -------------------------------------------------------


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    n: int
    successes: int
    success_rate: float
    cost_per_call: float
    cost_per_success: float | None   # None when there are 0 successes
    floor_breaches: int              # # task-types below the floor
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class TypeStat:
    arm: str
    task_type: str
    n: int
    successes: int
    success_rate: float
    cost_per_success: float | None


def _rate(successes: int, n: int) -> float:
    return successes / n if n else 0.0


def by_arm_type(calls: Iterable[ArmCall]) -> dict[tuple[str, str], TypeStat]:
    """Aggregate calls into (arm, task_type) cells."""
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for c in calls:
        cell = acc.setdefault((c.arm, c.task_type), {"n": 0, "s": 0, "succ_cost": 0.0})
        cell["n"] += 1
        if c.success:
            cell["s"] += 1
            cell["succ_cost"] += c.cost_usd
    out: dict[tuple[str, str], TypeStat] = {}
    for (arm, tt), v in acc.items():
        s = int(v["s"])
        out[(arm, tt)] = TypeStat(
            arm=arm,
            task_type=tt,
            n=int(v["n"]),
            successes=s,
            success_rate=round(_rate(s, int(v["n"])), 6),
            cost_per_success=round(v["succ_cost"] / s, 6) if s else None,
        )
    return out


def floor_by_type(
    cells: dict[tuple[str, str], TypeStat], floor_arm: str | None = None
) -> dict[str, float]:
    """The per-type success floor = the floor arm's measured success rate (A)."""
    floor_arm = floor_arm or config.FLOOR_ARM
    return {
        tt: stat.success_rate
        for (arm, tt), stat in cells.items()
        if arm == floor_arm
    }


def compute_arm_metrics(
    arm: str,
    calls: Iterable[ArmCall],
    floor: dict[str, float] | None = None,
) -> ArmMetrics:
    """Roll up one arm's calls. `floor` maps task_type -> floor success rate; a
    floor breach is a task-type where this arm trails the floor."""
    calls = [c for c in calls if c.arm == arm]
    n = len(calls)
    successes = sum(1 for c in calls if c.success)
    total_cost = sum(c.cost_usd for c in calls)
    succ_cost = sum(c.cost_usd for c in calls if c.success)
    rate = _rate(successes, n)
    lo, hi = wilson_ci(successes, n)

    breaches = 0
    if floor:
        cells = by_arm_type(calls)
        for (_, tt), stat in cells.items():
            f = floor.get(tt)
            if f is not None and stat.success_rate < f:
                breaches += 1

    return ArmMetrics(
        arm=arm,
        n=n,
        successes=successes,
        success_rate=round(rate, 6),
        cost_per_call=round(total_cost / n, 6) if n else 0.0,
        cost_per_success=round(succ_cost / successes, 6) if successes else None,
        floor_breaches=breaches,
        ci95_low=round(lo, 6),
        ci95_high=round(hi, 6),
    )


def compute_all(
    calls: Iterable[ArmCall], arms: Iterable[str] = ("A", "B", "C", "D")
) -> dict[str, ArmMetrics]:
    """Per-arm metrics for every arm, with the A-measured per-type floor applied."""
    calls = list(calls)
    floor = floor_by_type(by_arm_type(calls))
    return {a: compute_arm_metrics(a, calls, floor) for a in arms}


def traffic_weighted_success_rate(
    cells: dict[tuple[str, str], TypeStat],
    arm: str,
    weights: dict[str, float] | None = None,
) -> float:
    """Traffic-weighted success rate for one arm across task types. Default
    weights = each type's own n for that arm (so it equals the pooled rate)."""
    rows = [stat for (a, _), stat in cells.items() if a == arm]
    if not rows:
        return 0.0
    if weights is None:
        total_n = sum(r.n for r in rows)
        return round(sum(r.success_rate * r.n for r in rows) / total_n, 6) if total_n else 0.0
    total_w = sum(weights.get(r.task_type, 0.0) for r in rows)
    if not total_w:
        return 0.0
    return round(sum(r.success_rate * weights.get(r.task_type, 0.0) for r in rows) / total_w, 6)


# --- the pre-registered win check ------------------------------------------


@dataclass(frozen=True)
class WinVerdict:
    win: bool
    success_floor_met: bool          # C >= floor on every type
    success_within_epsilon: bool     # C.success >= A.success - eps
    cheaper_than_premium: bool       # C.cost_per_success < A.cost_per_success
    cheaper_than_panel: bool         # C.cost_per_success < D.cost_per_success
    drift_types: tuple[str, ...]     # types where C < floor (page founder)
    note: str


def win_check(
    calls: Iterable[ArmCall],
    *,
    epsilon_pt: float | None = None,
    floor_arm: str | None = None,
) -> WinVerdict:
    """Evaluate arm C against the pre-registered win conditions.

    epsilon_pt is in PERCENTAGE POINTS (default config.EPSILON_PT = 1.0).
    """
    calls = list(calls)
    eps = (config.EPSILON_PT if epsilon_pt is None else epsilon_pt) / 100.0
    cells = by_arm_type(calls)
    floor = floor_by_type(cells, floor_arm)
    m = {a: compute_arm_metrics(a, calls, floor) for a in ("A", "C", "D")}

    # C vs the per-type floor.
    c_cells = {tt: stat for (a, tt), stat in cells.items() if a == "C"}
    drift = tuple(
        sorted(tt for tt, f in floor.items() if tt in c_cells and c_cells[tt].success_rate < f)
    )
    floor_met = len(drift) == 0

    a_rate = m["A"].success_rate
    c_rate = m["C"].success_rate
    within_eps = c_rate >= (a_rate - eps)

    c_cps = m["C"].cost_per_success
    a_cps = m["A"].cost_per_success
    d_cps = m["D"].cost_per_success
    cheaper_premium = c_cps is not None and a_cps is not None and c_cps < a_cps
    cheaper_panel = c_cps is not None and d_cps is not None and c_cps < d_cps

    win = floor_met and within_eps and cheaper_premium and cheaper_panel
    return WinVerdict(
        win=win,
        success_floor_met=floor_met,
        success_within_epsilon=within_eps,
        cheaper_than_premium=cheaper_premium,
        cheaper_than_panel=cheaper_panel,
        drift_types=drift,
        note=(
            "C wins" if win else
            f"no-win: floor_met={floor_met} within_eps={within_eps} "
            f"cheaper_premium={cheaper_premium} cheaper_panel={cheaper_panel}"
        ),
    )
