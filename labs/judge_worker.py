"""judge_worker.py — Opus 4.7 sampler for the L14.2 daily cadence.

Reads `routing_outcomes` WHERE `judge_status='unlabeled'`. Samples per L14.2
spec (target ≥5%, cap 100/run; 20% cold-start when cell has <10 prior
labels). Calls Opus 4.7 via ainfera-inference (NOT direct Anthropic) and
writes `judge_score` (1.0-5.0) + `judge_rationale` + `judge_labeled_at`.

Self-firewall (Discipline #12): for the rows it labels in this run, the
worker EXCLUDES the Opus 4.7 model from the routable candidate set in the
ground-truth comparison — Opus cannot judge an outcome it produced.

Cost envelope: hard halt at cumulative $15/day. Alert via Slack #labs.

Inputs (env, Doppler-rendered):
    SUPABASE_URL                  postgres URL (read+write)
    AINFERA_API_KEY               labs-tenant key (routes Opus 4.7 via gateway)
    AINFERA_BASE_URL              https://api.ainfera.ai/v1
    LABS_SAMPLE_TARGET_PCT        0.05  (5%)
    LABS_SAMPLE_MAX_PER_RUN       100
    LABS_COLD_START_PCT_PER_CELL  0.20  (20% if cell <10 labels)
    LABS_COST_CAP_USD             15

Output: writes per-row judge_status='labeled' (or 'error'); returns a
JudgeRunSummary dataclass for the cron orchestrator to log.

This module is the LABS-side worker. It does NOT touch the gateway routing
itself; it only labels finished rows. The atomic policy publish on
PROMOTE is handled by `api/v1/admin/policy/publish` (W6-B).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# --- public types ----------------------------------------------------------


@dataclass(frozen=True)
class JudgeRunSummary:
    """Result of one judge_worker run (consumed by cron orchestrator)."""

    sampled_count: int
    labeled_count: int
    skipped_count: int     # cost-cap hit or row error
    cost_usd: float
    cells_touched: int     # distinct (task_type, candidate) cells
    halted_by: str | None  # "cost_cap" | "max_per_run" | None


# --- main entry -----------------------------------------------------------


def run_one_cycle(now_utc: Any = None) -> JudgeRunSummary:
    """Run one judge_worker cycle. Used by cron orchestrator + tests."""
    sample_target = float(os.environ.get("LABS_SAMPLE_TARGET_PCT", "0.05"))
    sample_max = int(os.environ.get("LABS_SAMPLE_MAX_PER_RUN", "100"))
    cold_start_pct = float(os.environ.get("LABS_COLD_START_PCT_PER_CELL", "0.20"))
    cost_cap = float(os.environ.get("LABS_COST_CAP_USD", "15"))

    log.info(
        "judge_worker: sample_target=%.2f sample_max=%d cold_start=%.2f cost_cap=$%.2f",
        sample_target, sample_max, cold_start_pct, cost_cap,
    )

    # Implementation phases — done in follow-up tickets, NOT in W6 scaffold:
    #   1. fetch unlabeled rows from routing_outcomes (DB query)
    #   2. stratified sample per cell (task_type, candidate); apply cold-start
    #   3. for each sampled row: call ainfera-inference with Opus 4.7 router hint
    #      (`model: auto` with quality_floor="high") on the prompt + response
    #      pair, returning judge_score 1-5 + rationale
    #   4. write back judge_score/rationale/labeled_at/judge_model
    #   5. track cumulative cost; halt if > cost_cap
    #   6. ensure Opus 4.7 is excluded from this row's candidate set when
    #      replay_gate runs (audit_event flag: judge_self_firewall_active)
    #
    # The W6 PR delivers this skeleton + the deterministic test harness in
    # tests/test_judge_worker.py. Real DB + Opus calls land in AIN-290
    # (capture+judge integration ticket).

    raise NotImplementedError(
        "judge_worker.run_one_cycle: skeleton only — real DB + Opus calls land "
        "in AIN-290 integration follow-up. W6 ships the deterministic test "
        "harness + cron orchestration only."
    )


# --- helpers (testable in isolation) ---------------------------------------


def select_sample(
    *,
    unlabeled_rows: list[dict[str, Any]],
    sample_target_pct: float,
    sample_max: int,
    cold_start_pct: float,
    cell_label_counts: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    """Stratified sample.

    Deterministic given the inputs (uses a stable hash of row id; no random).
    Cold-start: cells with <10 prior labels get sample_target = cold_start_pct.

    Returns the selected rows; never returns more than sample_max.
    """
    if not unlabeled_rows:
        return []

    selected: list[dict[str, Any]] = []
    for row in unlabeled_rows:
        cell = (row["task_type"], row["chosen_candidate"])
        prior = cell_label_counts.get(cell, 0)
        target = cold_start_pct if prior < 10 else sample_target_pct
        # Deterministic hash → reproducible sampling under CRN.
        h = abs(hash((row["id"], "labs-2026-05-28"))) % 10_000
        if h < int(target * 10_000):
            selected.append(row)
            if len(selected) >= sample_max:
                break
    return selected
