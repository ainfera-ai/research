"""AIN-542 Step 3 · Ainfera Council — pairwise verdicts + Dawid–Skene aggregation.

Tier B of the truth model: soft labels for the non-verifiable slice (the ~84% of
traffic the Tier-A anchor can't reach). Three properties make this trustworthy
rather than just a consensus:

1. **Pairwise + position-randomised.** Seats compare two candidate outputs (more
   reliable than pointwise 1–5) and each pair is judged in BOTH orders; the vote
   is the position-debiased agreement, cancelling position bias.
2. **Dawid–Skene EM aggregation, not majority vote.** Jointly estimates each
   seat's confusion matrix (reliability) and the latent label from the
   disagreement pattern — no gold required (Dawid & Skene 1979). Majority vote
   would assume equal reliability; DS down-weights unreliable seats.
3. **Self-preference family-exclusion** (in ``council_seats``) + **dispersion →
   Tier C**: high inter-seat disagreement flags a low-confidence verdict for
   deliberation / down-weighting.

This module is the AGGREGATION substance (pure, tested). The live seat model
calls are the seam (``SeatCaller``) wired in Step 3b on the :8646 Spark proxy;
calibration of DS reliability against the verifiable anchor (anchor-κ) is Step 4.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from labs.council_seats import COUNCIL_SEATS, Seat, eligible_seats


class Vote(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


VOTE_CLASSES: tuple[Vote, ...] = (Vote.A, Vote.B, Vote.TIE)

# A seat, shown an ordered pair, returns which POSITION it prefers (or a tie).
_POS = {"first", "second", "tie"}
# (Seat, first_output, second_output) -> 'first' | 'second' | 'tie'
SeatCaller = Callable[[Seat, str, str], str]


def _canon(pos: str, first_is: Vote, second_is: Vote) -> Vote:
    if pos == "first":
        return first_is
    if pos == "second":
        return second_is
    return Vote.TIE


def position_debiased_vote(pick_ab: Vote, pick_ba: Vote) -> Vote:
    """Combine a seat's canonical pick under order (A,B) and under order (B,A).
    Consistent across both orders → that candidate; flips with position (i.e. a
    position bias, or genuine indifference) → TIE."""
    if pick_ab is pick_ba and pick_ab in (Vote.A, Vote.B):
        return pick_ab
    return Vote.TIE


@dataclass(frozen=True)
class Comparison:
    """One A-vs-B item: the position-debiased eligible-seat votes + who was
    excluded by the family rule."""

    item_id: str
    family_a: str
    family_b: str
    seat_votes: dict[str, Vote]  # seat persona -> debiased canonical vote
    excluded_seats: list[str] = field(default_factory=list)


def make_comparison(
    item_id: str,
    output_a: str,
    output_b: str,
    family_a: str,
    family_b: str,
    seat_caller: SeatCaller,
    seats: tuple[Seat, ...] = COUNCIL_SEATS,
) -> Comparison:
    """Run the panel on one pair: apply family-exclusion, call each eligible seat
    in BOTH orders, debias. ``seat_caller`` is the live seam (Step 3b)."""
    eligible, excluded = eligible_seats({family_a, family_b}, seats)
    votes: dict[str, Vote] = {}
    for seat in eligible:
        ab = _canon(seat_caller(seat, output_a, output_b), Vote.A, Vote.B)
        ba = _canon(seat_caller(seat, output_b, output_a), Vote.B, Vote.A)
        votes[seat.persona] = position_debiased_vote(ab, ba)
    return Comparison(item_id, family_a, family_b, votes, [s.persona for s in excluded])


# ── Dawid–Skene EM (categorical, incomplete annotation matrix) ───────────────


@dataclass(frozen=True)
class DSResult:
    labels: dict[str, Vote]  # item -> argmax latent label
    label_probs: dict[str, dict[Vote, float]]  # item -> posterior over classes
    reliability: dict[str, float]  # seat -> expected accuracy E_k[π_kk]
    confusion: dict[str, list[list[float]]]  # seat -> KxK confusion matrix
    priors: list[float]  # class priors p_k
    classes: tuple[Vote, ...]


def dawid_skene(
    votes: dict[str, dict[str, Vote]],
    classes: tuple[Vote, ...] = VOTE_CLASSES,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    smoothing: float = 1e-9,
) -> DSResult:
    """Estimate latent labels + per-seat reliability from disagreement alone.

    ``votes``: ``{item_id: {seat: Vote}}`` (seats may be absent on an item —
    family-excluded or not run). No ground truth needed."""
    items = list(votes)
    seats = sorted({s for v in votes.values() for s in v})
    K = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}

    # init posterior T from vote fractions (majority-ish soft start)
    T: dict[str, list[float]] = {}
    for i in items:
        counts = [0.0] * K
        for lab in votes[i].values():
            counts[cidx[lab]] += 1.0
        tot = sum(counts) or 1.0
        T[i] = [c / tot for c in counts]

    p = [1.0 / K] * K
    pi: dict[str, list[list[float]]] = {
        s: [[1.0 / K] * K for _ in range(K)] for s in seats
    }
    prev_ll: float | None = None

    for _ in range(max_iter):
        # M-step
        n = len(items) or 1
        p = [sum(T[i][k] for i in items) / n for k in range(K)]
        num = {s: [[0.0] * K for _ in range(K)] for s in seats}
        den = {s: [0.0] * K for s in seats}
        for i in items:
            for s, lab in votes[i].items():
                li = cidx[lab]
                for k in range(K):
                    num[s][k][li] += T[i][k]
                    den[s][k] += T[i][k]
        for s in seats:
            for k in range(K):
                d = den[s][k]
                for li in range(K):
                    pi[s][k][li] = (
                        (num[s][k][li] + smoothing) / (d + K * smoothing)
                        if d > 0
                        else 1.0 / K
                    )

        # E-step (log space)
        ll = 0.0
        for i in items:
            log_un = [math.log(max(p[k], 1e-12)) for k in range(K)]
            for s, lab in votes[i].items():
                li = cidx[lab]
                for k in range(K):
                    log_un[k] += math.log(max(pi[s][k][li], 1e-12))
            m = max(log_un)
            exps = [math.exp(u - m) for u in log_un]
            z = sum(exps)
            T[i] = [e / z for e in exps]
            ll += m + math.log(z)

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    reliability = {s: sum(p[k] * pi[s][k][k] for k in range(K)) for s in seats}
    labels = {i: classes[max(range(K), key=lambda k: T[i][k])] for i in items}
    label_probs = {i: {classes[k]: T[i][k] for k in range(K)} for i in items}
    return DSResult(labels, label_probs, reliability, pi, p, classes)


def below_chance_seats(
    reliability: dict[str, float], n_classes: int = 3, margin: float = 0.0
) -> list[str]:
    """Seats whose expected accuracy is at/below chance (1/K) — candidates to
    filter. The DEFINITIVE filter validates reliability against the verifiable
    anchor (Step 4); this is the no-anchor screen."""
    chance = 1.0 / n_classes
    return sorted(s for s, r in reliability.items() if r <= chance + margin)


# ── verdict assembly ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CouncilVerdict:
    item_id: str
    label: Vote
    confidence: float  # posterior of the winning class
    dispersion: float  # normalized entropy of the posterior (high → Tier C)
    n_seats: int
    n_families: int
    excluded_seats: list[str]
    seat_votes: dict[str, Vote]

    @property
    def meets_floor(self) -> bool:
        """Step 3 acceptance: ≥3 seats from ≥2 families on this verdict."""
        return self.n_seats >= 3 and self.n_families >= 2


def _entropy(probs: dict[Vote, float]) -> float:
    ps = [p for p in probs.values() if p > 0]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(probs))  # normalize to [0,1]


def run_council(
    comparisons: list[Comparison],
    seats: tuple[Seat, ...] = COUNCIL_SEATS,
    *,
    dispersion_tier_c: float = 0.85,
) -> tuple[list[CouncilVerdict], DSResult]:
    """Aggregate a BATCH of comparisons: DS over the whole batch (so confusion
    matrices are estimable), then per-item verdict records. Returns the verdicts
    + the DS result (per-seat reliability for the batch)."""
    votes = {c.item_id: c.seat_votes for c in comparisons if c.seat_votes}
    ds = dawid_skene(votes) if votes else DSResult({}, {}, {}, {}, [], VOTE_CLASSES)
    persona_family = {s.persona: s.family for s in seats}
    verdicts: list[CouncilVerdict] = []
    for c in comparisons:
        probs = ds.label_probs.get(c.item_id, {})
        label = ds.labels.get(c.item_id, Vote.TIE)
        conf = probs.get(label, 0.0)
        fams = {persona_family.get(p, "?") for p in c.seat_votes}
        verdicts.append(
            CouncilVerdict(
                item_id=c.item_id,
                label=label,
                confidence=round(conf, 4),
                dispersion=round(_entropy(probs), 4) if probs else 1.0,
                n_seats=len(c.seat_votes),
                n_families=len(fams),
                excluded_seats=c.excluded_seats,
                seat_votes=c.seat_votes,
            )
        )
    return verdicts, ds


def needs_deliberation(verdict: CouncilVerdict, threshold: float = 0.85) -> bool:
    """Tier C trigger: dispersion above threshold → debate / down-weight."""
    return verdict.dispersion >= threshold
