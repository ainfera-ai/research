"""AIN-542 Step 4 · anchor-κ — calibrate the Council against the verifiable anchor.

The load-bearing mechanism that replaces human-gold. Run the Council on the
VERIFIABLE subset (where `verify()` IS ground truth) and score its verdicts
against `verify()`:

    anchor-κ = Gwet's AC1( Council label , verify-derived truth )   gate ≥ 0.60

Uses Gwet's AC1 (multi-rater, constitution lock 2026-06-20) instead of Cohen's
κ.  AC1 is more robust to prevalence imbalance — when one category dominates
(common in routing where most outputs pass), κ inflates while AC1 stays stable.

Also estimates per-seat anchor accuracy and flags **DS↔anchor divergence**: when
a seat's Dawid–Skene-predicted reliability (from agreement alone) diverges from
its anchor-measured accuracy, the seats' errors are correlated → consensus is
untrustworthy in that region → quarantine. That is how "the Council agrees but is
wrong" is caught with no human.

Pure calibration. The live cross-family seat calls are the seam
(`labs/seat_caller.gateway_seat_caller`); this module runs the Council via an
injected `SeatCaller` and computes the metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Callable

from labs.council import Comparison, Vote, make_comparison, run_council
from labs.council_seats import COUNCIL_SEATS, Seat

SeatCaller = Callable[[Seat, str, str, str], str]

ANCHOR_KAPPA_GATE = 0.60


@dataclass(frozen=True)
class AnchorPair:
    """A verifiable A-vs-B item: two outputs for the same prompt whose verify()
    rewards differ, so `truth` (the higher-verify output) is ground truth."""

    item_id: str
    output_a: str
    output_b: str
    family_a: str
    family_b: str
    truth: Vote  # Vote.A or Vote.B — which output verify() rates higher
    task: str = ""  # the request/question (AIN-546) so seats judge correctness


def build_pairs(rows: Sequence[dict[str, Any]]) -> list[AnchorPair]:
    """Form ground-truth pairs from verifiable rows. Each row needs
    ``{prompt_key, output_text, family, verify_reward}``. Within a prompt_key,
    pair a verify=1 output with a verify=0 output; the winner's side ALTERNATES
    (index parity, deterministic) so a position-biased Council can't score by
    always picking A."""
    pairs: list[AnchorPair] = []
    idx = 0
    keyed = sorted(rows, key=lambda r: str(r.get("prompt_key", "")))
    for key, grp in groupby(keyed, key=lambda r: str(r.get("prompt_key", ""))):
        group = list(grp)
        wins = [r for r in group if float(r.get("verify_reward", 0)) >= 1.0]
        losses = [r for r in group if float(r.get("verify_reward", 0)) <= 0.0]
        task = str(group[0].get("task", ""))  # same prompt → same task for the pair
        for w, ll in zip(wins, losses):
            iid = f"{key}:{idx}"
            if idx % 2 == 0:  # winner = A
                pairs.append(
                    AnchorPair(
                        iid,
                        w["output_text"],
                        ll["output_text"],
                        w["family"],
                        ll["family"],
                        Vote.A,
                        task,
                    )
                )
            else:  # winner = B
                pairs.append(
                    AnchorPair(
                        iid,
                        ll["output_text"],
                        w["output_text"],
                        ll["family"],
                        w["family"],
                        Vote.B,
                        task,
                    )
                )
            idx += 1
    return pairs


def gwet_ac1(pairs: Sequence[tuple[Vote, Vote]]) -> float | None:
    """Gwet's AC1 agreement coefficient for two raters over paired categorical
    labels (replaces Cohen's κ per constitution lock 2026-06-20).

    AC1 uses a different chance-agreement model than Cohen's κ:
      p_c = sum_k p_k * (1 - p_k) / (K - 1)
    where p_k is the marginal proportion for category k and K is the number of
    categories.  This makes AC1 more robust to prevalence imbalance (when one
    category dominates, κ artificially inflates while AC1 stays stable).

    For the two-rater, two-category case (Council vs verify-truth, A vs B):
      p_a = observed agreement = (# matches) / n
      p_c = (p_A*(1-p_A) + p_B*(1-p_B)) / (K-1)  where K=2
          = p_A*(1-p_A) + p_B*(1-p_B)  (since K-1 = 1)
          = 2 * p_A * p_B  (since p_B = 1 - p_A for two categories)
      AC1 = (p_a - p_c) / (1 - p_c)

    Returns None on empty; 1.0 on perfect single-label agreement (p_c=0 → AC1=1).
    """
    n = len(pairs)
    if n == 0:
        return None
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    K = len(labels)
    if K < 2:
        # Single label: everyone agrees → perfect agreement
        return 1.0
    # Observed agreement
    p_a = sum(1 for a, b in pairs if a == b) / n
    # Marginal proportions per category (averaged across both raters)
    marginals: dict[Vote, float] = {}
    for lab in labels:
        count_a = sum(1 for a, _ in pairs if a == lab)
        count_b = sum(1 for _, b in pairs if b == lab)
        marginals[lab] = (count_a + count_b) / (2 * n)
    # Gwet chance agreement: sum_k p_k*(1-p_k) / (K-1)
    p_c = sum(p * (1 - p) for p in marginals.values()) / (K - 1)
    if p_c >= 1.0:
        return 1.0
    return (p_a - p_c) / (1 - p_c)


def ds_anchor_divergence(
    ds_reliability: dict[str, float], anchor_accuracy: dict[str, float]
) -> dict[str, float]:
    """Per-seat |DS-predicted reliability − anchor-measured accuracy|. Large =
    correlated error (DS overconfident vs reality) → quarantine that seat/region."""
    return {
        s: abs(ds_reliability.get(s, 0.0) - anchor_accuracy[s]) for s in anchor_accuracy
    }


@dataclass(frozen=True)
class AnchorKappaResult:
    kappa: float | None
    n_pairs: int
    council_accuracy: float
    per_seat_anchor_accuracy: dict[str, float]
    ds_reliability: dict[str, float]
    divergence: dict[str, float]
    eligible: bool  # kappa ≥ gate
    max_divergence: float = field(default=0.0)
    quarantined_seats: list[str] = field(default_factory=list)


def quarantine_seats(
    result: AnchorKappaResult,
    *,
    min_anchor_accuracy: float = 0.5,
    max_overconfidence: float = 0.4,
) -> list[str]:
    """Seats to QUARANTINE (Step 4 / §3). DIRECTIONAL — only drop a seat when it is
    actually unreliable by the verifiable anchor:

      1. anchor accuracy ≤ chance (below-chance → bad), OR
      2. the panel OVER-trusts it: ``DS_reliability − anchor_accuracy >
         max_overconfidence`` (DS thinks it's reliable but the anchor says it's
         not — the dangerous correlated-error case §3 targets).

    NOT the symmetric ``|DS − anchor|``: a GOOD seat that DS merely UNDER-rates
    (high anchor accuracy, low DS because it abstains a lot) is harmless and must
    be KEPT — quarantining it (the scale-run bug) throws away a reliable judge."""
    bad: set[str] = set()
    for seat, acc in result.per_seat_anchor_accuracy.items():
        if acc <= min_anchor_accuracy:
            bad.add(seat)
        elif result.ds_reliability.get(seat, 0.0) - acc > max_overconfidence:
            bad.add(seat)
    return sorted(bad)


def _score(
    comparisons: list[Comparison],
    pairs: Sequence[AnchorPair],
    seats: tuple[Seat, ...],
    gate: float,
    quarantined: list[str],
) -> AnchorKappaResult:
    verdicts, ds = run_council(comparisons, seats)
    truth = {p.item_id: p.truth for p in pairs}
    decided = [
        (v.label, truth[v.item_id]) for v in verdicts if v.label in (Vote.A, Vote.B)
    ]
    kappa = gwet_ac1(decided)
    accuracy = (sum(1 for a, b in decided if a == b) / len(decided)) if decided else 0.0

    seat_acc: dict[str, float] = {}
    for seat in seats:
        hits = tot = 0
        for comp in comparisons:
            vote = comp.seat_votes.get(seat.persona)
            if vote in (Vote.A, Vote.B):
                tot += 1
                hits += int(vote == truth[comp.item_id])
        if tot:
            seat_acc[seat.persona] = hits / tot

    divergence = ds_anchor_divergence(ds.reliability, seat_acc)
    return AnchorKappaResult(
        kappa=kappa,
        n_pairs=len(pairs),
        council_accuracy=round(accuracy, 4),
        per_seat_anchor_accuracy={s: round(a, 4) for s, a in seat_acc.items()},
        ds_reliability={s: round(r, 4) for s, r in ds.reliability.items()},
        divergence={s: round(d, 4) for s, d in divergence.items()},
        eligible=(kappa is not None and kappa >= gate),
        max_divergence=round(max(divergence.values()), 4) if divergence else 0.0,
        quarantined_seats=list(quarantined),
    )


def compute_anchor_kappa(
    pairs: Sequence[AnchorPair],
    seat_caller: SeatCaller,
    seats: tuple[Seat, ...] = COUNCIL_SEATS,
    *,
    gate: float = ANCHOR_KAPPA_GATE,
    reliability_filter: bool = True,
    min_anchor_accuracy: float = 0.5,
    max_overconfidence: float = 0.4,
) -> AnchorKappaResult:
    """Run the Council on the verifiable pairs and score against verify(). With
    ``reliability_filter`` (Step 4): score once, quarantine anchor-unreliable seats,
    then RE-AGGREGATE without them (no new model calls — DS re-runs over the kept
    votes). The returned result reflects the filtered panel + records who was
    quarantined."""
    comparisons = [
        make_comparison(
            p.item_id,
            p.task,
            p.output_a,
            p.output_b,
            p.family_a,
            p.family_b,
            seat_caller,
            seats,
        )
        for p in pairs
    ]
    first = _score(comparisons, pairs, seats, gate, [])
    if not reliability_filter:
        return first
    bad = quarantine_seats(
        first,
        min_anchor_accuracy=min_anchor_accuracy,
        max_overconfidence=max_overconfidence,
    )
    if not bad:
        return first
    # re-aggregate: drop the quarantined seats' votes + the seats from the panel
    bad_set = set(bad)
    filtered = [
        Comparison(
            c.item_id,
            c.family_a,
            c.family_b,
            {p: v for p, v in c.seat_votes.items() if p not in bad_set},
            c.excluded_seats,
        )
        for c in comparisons
    ]
    kept = tuple(s for s in seats if s.persona not in bad_set)
    return _score(filtered, pairs, kept, gate, bad)


# Pull the verifiable subset (already verify-scored by Step 2c) for pairing.
# prompt_key groups outputs to the same task; the runner binds %(days)s.
ANCHOR_PAIRS_SQL = (
    "SELECT ro.id, ro.task_type, ro.chosen_model_slug, ro.reward AS verify_reward, "
    "       md5(i.request_payload::text) AS prompt_key, "
    "       COALESCE(i.response_payload #>> '{choices,0,message,content}', "
    "                i.response_payload #>> '{content,0,text}') AS output_text "
    "FROM routing_outcomes ro "
    "JOIN inferences i ON i.id = ro.inference_id "
    "WHERE ro.reward_source = 'verify' "
    "  AND NOT ro.exclude_from_training "
    "  AND ro.created_at >= now() - (%(days)s || ' days')::interval "
    "ORDER BY prompt_key"
)


# Backward compatibility: cohen_kappa is now gwet_ac1 (constitution lock
# 2026-06-20). Existing imports of cohen_kappa still work but use AC1.
cohen_kappa = gwet_ac1
