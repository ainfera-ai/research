"""AIN-542 Step 3 · Ainfera Council seat roster (cross-family panel / PoLL).

A panel of disjoint-family judges beats a single large judge (Verga et al.,
arXiv:2404.18796): less intra-model bias, no single-family self-preference. Seats
are real, active catalog models (verified live, dftfpwzqxoebwzepygzl) spanning
≥5 disjoint maker families, mapped to Aratar personas.

Two hard rules this module encodes:

1. **Self-preference family-exclusion.** When a candidate output came from family
   F, every family-F seat is removed from that verdict (LLM judges recognise and
   over-score their own family — Panickssery et al., arXiv:2404.13076). A pairwise
   A-vs-B comparison excludes ≤2 families, so a 5-disjoint-family roster always
   leaves ≥3 seats / ≥2 families — the Step 3 acceptance floor.

2. **Spark placement (cost).** Open-weight seats run ON Spark Labs (zero API
   cost); frontier seats are API and reserved for the calibration / tie-break
   subset, not every verdict. ``on_spark`` records the split for the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Seat:
    persona: str  # Aratar persona (display)
    model_slug: str  # real catalog slug (must be active + routable)
    family: str  # maker family — the self-preference exclusion key
    on_spark: bool  # True => open-weight, run locally on Spark Labs
    role: str = "seat"  # 'chair' (aggregates) | 'dissent' (adversarial) | 'seat'


# The core cross-family panel. 5 disjoint families → a pairwise verdict (excludes
# ≤2 families) always retains ≥3 seats / ≥2 families. AA-index in comments (live).
COUNCIL_SEATS: tuple[Seat, ...] = (
    Seat("Námo", "claude-opus-4-7", "anthropic", on_spark=False, role="chair"),  # aa 73
    Seat("Manwë", "gpt-5-5", "openai", on_spark=False, role="seat"),  # aa 70
    Seat("Aulë", "gemini-3-1-pro", "google", on_spark=False, role="seat"),  # aa 68
    Seat("Tulkas", "grok-4", "xai", on_spark=False, role="dissent"),  # aa 65
    Seat(
        "Yavanna", "llama-4-405b-together", "meta", on_spark=True, role="seat"
    ),  # aa 62, open-weight
)

# Extended open-weight pool — additional Spark seats to deepen family diversity at
# zero API cost (loaded/unloaded sequentially in the nightly batch). Wired in 3b.
SPARK_SEAT_POOL: tuple[Seat, ...] = (
    Seat("Ulmo", "mistral-large-3", "mistral", on_spark=True, role="seat"),  # aa 60
    Seat(
        "Oromë", "qwen-3-7-max-together", "alibaba", on_spark=True, role="seat"
    ),  # aa 57
    Seat("Vairë", "minimax-m3-novita", "minimax", on_spark=True, role="seat"),  # aa 55
    Seat(
        "Estë", "deepseek-v4-pro-deepinfra", "deepseek", on_spark=True, role="seat"
    ),  # aa 52
)

_MIN_FAMILIES = 5


def _validate_roster(seats: tuple[Seat, ...]) -> None:
    families = {s.family for s in seats}
    if len(families) < _MIN_FAMILIES:
        raise ValueError(
            f"roster spans {len(families)} families, need ≥{_MIN_FAMILIES} disjoint"
        )
    if len({s.model_slug for s in seats}) != len(seats):
        raise ValueError("duplicate model_slug in roster")


_validate_roster(COUNCIL_SEATS)


def eligible_seats(
    candidate_families: set[str], seats: tuple[Seat, ...] = COUNCIL_SEATS
) -> tuple[list[Seat], list[Seat]]:
    """Self-preference family-exclusion. Returns ``(eligible, excluded)``: a seat
    is excluded iff its family is one of the candidate families being judged."""
    eligible = [s for s in seats if s.family not in candidate_families]
    excluded = [s for s in seats if s.family in candidate_families]
    return eligible, excluded


def families_of(seats: list[Seat]) -> set[str]:
    return {s.family for s in seats}
