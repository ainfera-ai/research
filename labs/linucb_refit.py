"""linucb_refit.py — daily bandit refit over the rolling-30d labeled corpus.

Reads labeled `routing_outcomes` from the last 30 days; refits per-cell
`q_empirical` via LinUCB; enforces a ≥5% exploration floor on every cell.
Outputs a versioned `policy_v<YYYYMMDD>-NNN.json` artifact that
`replay_gate.py` consumes for the PROMOTE/HOLD decision.

LinUCB confidence interval (per task_type × candidate cell):
    UCB = q_empirical + alpha * sqrt(ln(t) / n_cell)
where:
    q_empirical = mean(reward) for the cell (reward = judge_score normalized 0-1)
    alpha       = exploration parameter (default 1.0; tunable via env)
    t           = total labeled rows across all cells
    n_cell      = labeled rows in this cell
The exploration floor ensures even high-q cells stay sampled enough to
detect drift.

CRN seeding: every run uses LABS_CRN_SEED if present (deterministic across
re-runs for the same calendar day). This is what makes the daily delta
honest — the bandit state is reproducible from the same inputs.

Output schema (policy_v<YYYYMMDD>-NNN.json):
    {
      "version": "v20260602-001",
      "computed_at": "2026-06-02T20:30:00Z",
      "input_corpus": {"start": "...", "end": "...", "n_rows": 1234, "cells": 42},
      "exploration_floor_pct": 0.05,
      "alpha": 1.0,
      "cells": [
        {"task_type": "...", "candidate": "...", "q_empirical": 0.78,
         "n_labeled": 156, "ucb": 0.83, "explore_pct": 0.05}
      ]
    }

References:
  ainfera-vault methodology/q_empirical-v1.3.md
  ainfera-vault methodology/daily-training-cadence.md §"LinUCB refit"
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellEstimate:
    task_type: str
    candidate: str
    q_empirical: float
    n_labeled: int
    ucb: float
    explore_pct: float


@dataclass
class PolicyCandidate:
    version: str
    computed_at: str
    input_corpus: dict[str, Any]
    exploration_floor_pct: float
    alpha: float
    cells: list[CellEstimate] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "computed_at": self.computed_at,
                "input_corpus": self.input_corpus,
                "exploration_floor_pct": self.exploration_floor_pct,
                "alpha": self.alpha,
                "cells": [c.__dict__ for c in self.cells],
            },
            indent=2,
            sort_keys=True,
        )


def fit(
    labeled_rows: list[dict[str, Any]],
    *,
    seed: int | None = None,
    alpha: float = 1.0,
    exploration_floor_pct: float = 0.05,
    today: datetime | None = None,
) -> PolicyCandidate:
    """Pure-function LinUCB refit. Deterministic given (rows, seed, alpha).

    CRN-safe: same inputs → same output bytes.
    """
    if seed is None:
        seed = int(os.environ.get("LABS_CRN_SEED", "20260528"))
    rng = random.Random(seed)
    today = today or datetime.now(tz=timezone.utc)

    # Aggregate per cell (task_type, candidate)
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for row in labeled_rows:
        key = (row["task_type"], row["chosen_candidate"])
        c = cells.setdefault(key, {"n": 0, "reward_sum": 0.0})
        c["n"] += 1
        # judge_score 1-5 → reward 0-1 (linear scale)
        c["reward_sum"] += (row["judge_score"] - 1.0) / 4.0

    t = max(1, sum(c["n"] for c in cells.values()))

    estimates: list[CellEstimate] = []
    for (tt, cand), c in sorted(cells.items()):
        n = max(1, int(c["n"]))
        q = c["reward_sum"] / n
        ucb = q + alpha * math.sqrt(math.log(t) / n)
        # Exploration floor: every cell gets at least floor_pct allocation.
        # The actual sampling is the gateway router's job; this just
        # publishes the floor as part of the policy.
        explore_pct = max(exploration_floor_pct, 1.0 / (n + 1))
        estimates.append(
            CellEstimate(
                task_type=tt,
                candidate=cand,
                q_empirical=round(q, 6),
                n_labeled=int(c["n"]),
                ucb=round(ucb, 6),
                explore_pct=round(explore_pct, 6),
            )
        )

    # Tiny entropy nudge via RNG (deterministic via seed) — placeholder for
    # the production tie-break rule.
    rng.shuffle(estimates)
    estimates.sort(key=lambda c: c.ucb, reverse=True)

    version_base = today.strftime("%Y%m%d")
    return PolicyCandidate(
        version=f"v{version_base}-001",
        computed_at=today.isoformat(timespec="seconds"),
        input_corpus={
            "n_rows": len(labeled_rows),
            "cells": len(cells),
        },
        exploration_floor_pct=exploration_floor_pct,
        alpha=alpha,
        cells=estimates,
    )
