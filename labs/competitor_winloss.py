"""AIN-376 · Tier-0 competitor win/loss matrix + AIQ (offline, Spark Labs).

PURE aggregation + blinding; no I/O, no network. The weekly Spark batch routes one identical
eval set through each competitor router (OpenRouter / Martian / Not Diamond / Bedrock IPR)
AND through ainfera-inference, BLIND-judges every output (the judge must not know which is
ours — untreated self-enhancement bias makes the intel worthless), and feeds the per-request
verdicts here to produce the win/loss matrix + AIQ + cost-quality Pareto points by task class.

Conventions follow RouterBench / RouteLLM so results are academically comparable:
    AIQ (Average Improvement in Quality) = mean(q_ainfera - q_competitor).

FENCE (AIN-375): this consumes blind-judged pairs and emits a matrix; it writes only to
``competitor_probes``, NEVER ``routing_outcomes`` / the training/distill moat. Spark Labs
tenant only.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgedPair:
    """One blind-judged head-to-head on an identical request."""

    task_class: str
    competitor: str  # competitor router slug (the eval target)
    q_ainfera: float  # blind-judged quality in [0,1]
    q_competitor: float
    cost_ainfera_usd: float = 0.0
    cost_competitor_usd: float = 0.0


@dataclass(frozen=True)
class CellWinLoss:
    """Win/loss + AIQ + Pareto means for one (task_class, competitor) cell."""

    task_class: str
    competitor: str
    n: int
    wins: int  # ainfera quality strictly higher
    losses: int
    ties: int
    win_rate: float  # wins / (wins + losses); 0.0 when all ties
    aiq: float  # mean(q_ainfera - q_competitor) — the headline metric
    q_ainfera_mean: float
    q_competitor_mean: float
    cost_ainfera_mean: float
    cost_competitor_mean: float


def blind_pair(
    ainfera_output: str,
    competitor_output: str,
    *,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, str]]:
    """Randomly assign the two outputs to anonymous slots A/B so the judge cannot tell which
    is ainfera (self-enhancement-bias control). Returns (blinded, mapping) where blinded is
    ``{"A": ..., "B": ...}`` and mapping is ``{"A": "ainfera"|"competitor", ...}`` for
    de-blinding the verdict afterwards. Deterministic given ``rng``."""
    if rng.random() < 0.5:
        return {"A": ainfera_output, "B": competitor_output}, {"A": "ainfera", "B": "competitor"}
    return {"A": competitor_output, "B": ainfera_output}, {"A": "competitor", "B": "ainfera"}


def aggregate_winloss(pairs: list[JudgedPair], *, tie_eps: float = 1e-9) -> list[CellWinLoss]:
    """Aggregate blind-judged pairs into the per-(task_class, competitor) win/loss matrix.

    A win = ainfera quality strictly above the competitor by more than ``tie_eps``; loss =
    strictly below; else tie. AIQ = mean quality delta. Sorted (task_class, competitor) for a
    deterministic, replayable matrix. Empty input ⇒ empty matrix."""
    groups: dict[tuple[str, str], list[JudgedPair]] = defaultdict(list)
    for p in pairs:
        groups[(p.task_class, p.competitor)].append(p)

    out: list[CellWinLoss] = []
    for (task_class, competitor), ps in sorted(groups.items()):
        n = len(ps)
        wins = sum(1 for p in ps if p.q_ainfera - p.q_competitor > tie_eps)
        losses = sum(1 for p in ps if p.q_competitor - p.q_ainfera > tie_eps)
        ties = n - wins - losses
        decisive = wins + losses
        out.append(
            CellWinLoss(
                task_class=task_class,
                competitor=competitor,
                n=n,
                wins=wins,
                losses=losses,
                ties=ties,
                win_rate=round(wins / decisive, 6) if decisive else 0.0,
                aiq=round(sum(p.q_ainfera - p.q_competitor for p in ps) / n, 6),
                q_ainfera_mean=round(sum(p.q_ainfera for p in ps) / n, 6),
                q_competitor_mean=round(sum(p.q_competitor for p in ps) / n, 6),
                cost_ainfera_mean=round(sum(p.cost_ainfera_usd for p in ps) / n, 8),
                cost_competitor_mean=round(sum(p.cost_competitor_usd for p in ps) / n, 8),
            )
        )
    return out


__all__ = ["CellWinLoss", "JudgedPair", "aggregate_winloss", "blind_pair"]
