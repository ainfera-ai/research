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

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from labs.shrinkage import shrinkage_posterior
from labs.thompson import Posterior, allocate_with_floor, thompson_probabilities

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellEstimate:
    task_type: str
    candidate: str
    q_empirical: float
    n_labeled: int
    ucb: float
    explore_pct: float
    # D1 shrinkage audit (AIN-542): the benchmark prior blended in, and the fraction
    # of q_empirical it still contributes (decays with n). None/0.0 when shrinkage is
    # off (prior_strength=0) — see _cell_json, which omits them so the v0 artifact is
    # byte-identical.
    q_prior: float | None = None
    prior_weight: float = 0.0
    # D7 Thompson allocation (AIN-542): this cell's exploration share within its
    # task_type, from the posterior P[best] + a min-sample floor. None when Thompson
    # is off — _cell_json omits it so the v0 artifact stays byte-identical.
    alloc_weight: float | None = None


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
                "cells": [_cell_json(c) for c in self.cells],
            },
            indent=2,
            sort_keys=True,
        )


def _cell_json(c: CellEstimate) -> dict[str, Any]:
    """Serialise a cell. The D1 shrinkage fields (q_prior, prior_weight) appear ONLY
    when shrinkage was active for the cell, so a no-shrinkage refit emits the exact
    v0 schema → CRN replay across the cutover stays byte-identical."""
    d: dict[str, Any] = {
        "task_type": c.task_type,
        "candidate": c.candidate,
        "q_empirical": c.q_empirical,
        "n_labeled": c.n_labeled,
        "ucb": c.ucb,
        "explore_pct": c.explore_pct,
    }
    if c.q_prior is not None:
        d["q_prior"] = c.q_prior
        d["prior_weight"] = c.prior_weight
    if c.alloc_weight is not None:
        d["alloc_weight"] = c.alloc_weight
    return d


def fit(
    labeled_rows: list[dict[str, Any]],
    *,
    seed: int | None = None,
    alpha: float = 1.0,
    exploration_floor_pct: float = 0.05,
    today: datetime | None = None,
    reward_fn: Callable[[dict[str, Any]], float] | None = None,
    priors: dict[str, float] | None = None,
    prior_strength: float = 0.0,
) -> PolicyCandidate:
    """Pure-function LinUCB refit. Deterministic given (rows, seed, alpha).

    CRN-safe: same inputs → same output bytes.

    ``reward_fn`` maps a row → reward in [0,1]. Default = the judge mapping
    (``(judge_score-1)/4``) for backward compatibility. The judge-FREE path
    (``labeled_corpus.assemble_corpus``) pre-computes a ``reward`` field and
    passes ``reward_fn=lambda r: r["reward"]`` so no judge label is required.

    D1 shrinkage (AIN-542): when ``prior_strength > 0`` and ``priors`` carries a
    benchmark q_prior for a candidate, that cell's ``q_empirical`` becomes the
    empirical-Bayes posterior ``(prior_strength·prior + Σreward)/(prior_strength+n)``
    instead of the raw mean — the prior decays as labeled evidence accrues. Default
    ``prior_strength=0`` ⇒ raw mean ⇒ byte-identical to v0.
    """
    if reward_fn is None:
        reward_fn = lambda r: (r["judge_score"] - 1.0) / 4.0  # noqa: E731
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
        c["reward_sum"] += reward_fn(row)

    t = max(1, sum(c["n"] for c in cells.values()))

    estimates: list[CellEstimate] = []
    priors = priors or {}
    for (tt, cand), c in sorted(cells.items()):
        n = max(1, int(c["n"]))
        # D1 shrinkage: blend the benchmark prior with the empirical mean (the prior
        # decays as n grows). Active only when prior_strength>0 AND this candidate has
        # a prior — otherwise prior_strength=0 ⇒ q = raw mean (v0, byte-identical).
        prior_for_cell = priors.get(cand) if prior_strength > 0 else None
        ps = prior_strength if prior_for_cell is not None else 0.0
        est = shrinkage_posterior(
            prior_for_cell if prior_for_cell is not None else 0.0,
            c["reward_sum"],
            n,
            prior_strength=ps,
        )
        q = est.q_posterior
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
                q_prior=round(prior_for_cell, 6)
                if prior_for_cell is not None
                else None,
                prior_weight=round(est.prior_weight, 6)
                if prior_for_cell is not None
                else 0.0,
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


def apply_thompson_allocation(
    policy: PolicyCandidate,
    *,
    prior_strength: float = 0.0,
    min_samples: int = 30,
    floor_pct: float = 0.05,
    seed: int | None = None,
    draws: int = 4000,
) -> PolicyCandidate:
    """Return a copy of `policy` with per-cell ``alloc_weight`` set by Thompson
    sampling + a min-sample floor, computed WITHIN each task_type (D7, AIN-542).

    The Beta posterior per cell is recovered from its published ``q_empirical`` and
    ``n_labeled`` given ``prior_strength`` — the same value passed to ``fit`` — since
    α+β = prior_strength+n and α = q_empirical·(α+β). This is a pure read of the
    policy artifact (no corpus needed) and is deterministic given ``seed`` (defaults
    to LABS_CRN_SEED), so the allocation replays exactly.

    A separate post-processor (not folded into ``fit``) so a policy that does NOT go
    through it is byte-identical to v0 — Thompson is strictly opt-in.
    """
    if seed is None:
        seed = int(os.environ.get("LABS_CRN_SEED", "20260528"))

    by_task: dict[str, list[CellEstimate]] = {}
    for c in policy.cells:
        by_task.setdefault(c.task_type, []).append(c)

    alloc_by_cell: dict[tuple[str, str], float] = {}
    for task_offset, (tt, group) in enumerate(sorted(by_task.items())):
        posteriors: list[Posterior] = []
        n_by: dict[str, int] = {}
        for c in group:
            sn = prior_strength + c.n_labeled
            alpha = max(c.q_empirical * sn, 1e-9)
            beta = max((1.0 - c.q_empirical) * sn, 1e-9)
            posteriors.append(Posterior(c.candidate, alpha, beta, c.n_labeled))
            n_by[c.candidate] = c.n_labeled
        # distinct per-task seed so independent task_types don't share a draw stream
        probs = thompson_probabilities(posteriors, draws=draws, seed=seed + task_offset)
        alloc = allocate_with_floor(
            probs, n_by, min_samples=min_samples, floor_pct=floor_pct
        )
        for c in group:
            alloc_by_cell[(tt, c.candidate)] = alloc[c.candidate]

    new_cells = [
        replace(c, alloc_weight=round(alloc_by_cell[(c.task_type, c.candidate)], 6))
        for c in policy.cells
    ]
    return replace(policy, cells=new_cells)


def refit_policy(
    labeled_rows: list[dict[str, Any]],
    *,
    seed: int | None = None,
    alpha: float = 1.0,
    exploration_floor_pct: float = 0.05,
    today: datetime | None = None,
    reward_fn: Callable[[dict[str, Any]], float] | None = None,
    q_priors: dict[str, float] | None = None,
    prior_strength: float = 0.0,
    thompson: bool = False,
    min_samples: int = 30,
    thompson_floor_pct: float = 0.05,
    thompson_draws: int = 4000,
) -> PolicyCandidate:
    """The nightly refit's composition point: ``fit`` (D1 shrinkage when
    ``prior_strength>0``) then optionally ``apply_thompson_allocation`` (D7 when
    ``thompson``). With ``prior_strength=0`` and ``thompson=False`` this is exactly
    ``fit`` — byte-identical to v0. Both knobs are env-gated at the CLI boundary."""
    policy = fit(
        labeled_rows,
        seed=seed,
        alpha=alpha,
        exploration_floor_pct=exploration_floor_pct,
        today=today,
        reward_fn=reward_fn,
        priors=q_priors,
        prior_strength=prior_strength,
    )
    if thompson:
        policy = apply_thompson_allocation(
            policy,
            prior_strength=prior_strength,
            min_samples=min_samples,
            floor_pct=thompson_floor_pct,
            seed=seed,
            draws=thompson_draws,
        )
    return policy


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _main(argv: list[str] | None = None) -> int:
    """`python3 -m labs.linucb_refit` — refit a policy candidate from a labeled-corpus
    JSON file. D1/D7 are env-gated and DEFAULT OFF (so the published candidate is the v0
    policy until the founder sets the flags):

        LABS_PRIOR_STRENGTH     D1 shrinkage prior pseudo-count (default 0 = raw mean)
        LABS_THOMPSON           D7 Thompson allocation on/off (default off)
        LABS_MIN_SAMPLES        D7 min-sample floor count (default 30)
        LABS_THOMPSON_FLOOR_PCT D7 floor share per under-sampled cell (default 0.05)
        LABS_CRN_SEED           CRN seed (existing)

    The corpus + priors come in as files — labs stays DB-free, so the ops wrapper binds
    `labeled_corpus.select_labeled_corpus_sql()` → --corpus and the catalog q_priors →
    --priors. The candidate then flows through replay_gate (cron.sh) → founder promote.
    """
    p = argparse.ArgumentParser(prog="labs.linucb_refit")
    p.add_argument(
        "--corpus", required=True, help="JSON: [{task_type, chosen_candidate, reward}]"
    )
    p.add_argument(
        "--output", required=True, help="path to write the policy-candidate JSON"
    )
    p.add_argument(
        "--priors", help="JSON: {candidate_slug: q_prior} for D1 shrinkage (optional)"
    )
    p.add_argument("--date", help="policy date YYYY-MM-DD (default: today UTC)")
    p.add_argument(
        "--input-corpus-days",
        type=int,
        default=30,
        help="recorded for provenance; corpus binding itself is the ops wrapper's job",
    )
    args = p.parse_args(argv)

    rows = json.loads(Path(args.corpus).read_text())
    q_priors = json.loads(Path(args.priors).read_text()) if args.priors else None
    today = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.date
        else datetime.now(tz=timezone.utc)
    )

    prior_strength = float(os.environ.get("LABS_PRIOR_STRENGTH", "0"))
    thompson = _truthy(os.environ.get("LABS_THOMPSON"))
    min_samples = int(os.environ.get("LABS_MIN_SAMPLES", "30"))
    floor_pct = float(os.environ.get("LABS_THOMPSON_FLOOR_PCT", "0.05"))

    policy = refit_policy(
        rows,
        today=today,
        reward_fn=lambda r: r["reward"],
        q_priors=q_priors,
        prior_strength=prior_strength,
        thompson=thompson,
        min_samples=min_samples,
        thompson_floor_pct=floor_pct,
    )
    Path(args.output).write_text(policy.to_json())
    log.info(
        "refit → %s (cells=%d prior_strength=%s thompson=%s)",
        args.output,
        len(policy.cells),
        prior_strength,
        thompson,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_main())
