"""AIN-561 · distilled-judge cost report — API calls + teacher $ avoided per nightly run.

The owned student judge (LoRA on Spark) replaces frontier-teacher (Opus) API calls on the
bulk-labeling path 1:1. This reports what each nightly labeling run AVOIDS: teacher API
calls and their dollar cost, net of the cheap local student inference.

Screening-only — this is an OPERATIONAL cost report, never a moat/reward signal and never an
input to routing or promotion. Pure + parameterized: the per-call unit prices are inputs
(no hardcoded model rates), so a catalog reprice can never silently rewrite a past run's
report, and the same inputs always produce the same report (CRN-stable).
"""

from __future__ import annotations

from dataclasses import dataclass

_USD = 6  # dollar rounding (micro-dollar) — matches the api ledger's Numeric(18,6)


@dataclass(frozen=True)
class DistillCostReport:
    """One nightly run's API-calls-/cost-avoided summary."""

    run_date: str
    items_labeled: int  # verdicts the student produced this run
    teacher_calls_avoided: int  # teacher API calls the student replaced (1:1 on the bulk path)
    teacher_unit_usd: float  # teacher (Opus) $ per labeling call
    student_unit_usd: float  # student (local Spark) $ per labeling call (~0)
    teacher_cost_avoided_usd: float  # what an all-teacher run would have cost
    student_cost_incurred_usd: float  # what the student run actually cost
    net_cost_avoided_usd: float  # teacher_cost_avoided - student_cost_incurred
    net_cost_avoided_pct: float  # vs the all-teacher baseline; 0.0 when the baseline is 0


def build_cost_report(
    *,
    run_date: str,
    items_labeled: int,
    teacher_unit_usd: float,
    student_unit_usd: float,
) -> DistillCostReport:
    """Compute the cost-avoided report for one labeling run.

    The student replaces the teacher 1:1 on the bulk-labeling path, so
    ``teacher_calls_avoided == items_labeled``. Net avoided = the teacher cost that run
    would have incurred minus the student cost it actually incurred. ``pct`` is floored to
    0.0 when there is no baseline (zero items) — never a division by zero.
    """
    if items_labeled < 0:
        raise ValueError("items_labeled must be >= 0")
    if teacher_unit_usd < 0 or student_unit_usd < 0:
        raise ValueError("unit costs must be >= 0")

    teacher_cost_avoided = items_labeled * teacher_unit_usd
    student_cost = items_labeled * student_unit_usd
    net = teacher_cost_avoided - student_cost
    pct = (100.0 * net / teacher_cost_avoided) if teacher_cost_avoided > 0 else 0.0

    return DistillCostReport(
        run_date=run_date,
        items_labeled=items_labeled,
        teacher_calls_avoided=items_labeled,
        teacher_unit_usd=teacher_unit_usd,
        student_unit_usd=student_unit_usd,
        teacher_cost_avoided_usd=round(teacher_cost_avoided, _USD),
        student_cost_incurred_usd=round(student_cost, _USD),
        net_cost_avoided_usd=round(net, _USD),
        net_cost_avoided_pct=round(pct, 2),
    )
