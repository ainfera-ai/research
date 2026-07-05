"""AIN-561 · distilled-judge ship gate — agreement vs the Council anchor (offline, Spark Labs).

The LoRA distill (teacher + Council → an owned local judge, AIN-304) trains on the DGX Spark
GPU; THIS is the gate that decides whether the distilled judge may ship as the bulk labeler.
It must AGREE with the Council anchor on a held-out set — **never assumed**. A distilled judge
below the bar would silently corrupt the reward signal the whole loop learns from, so it does
not ship. Reuses the project's Cohen's κ (``labs.anchor_kappa.cohen_kappa``) and the same
≥0.60 bar as the anchor-κ gate (Landis–Koch "substantial").

PURE; no I/O, no GPU. The distill/training is the Spark seam; this scores the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from labs.anchor_kappa import ANCHOR_KAPPA_GATE, cohen_kappa


@dataclass(frozen=True)
class DistillAgreement:
    n: int
    raw_agreement: float  # Po — fraction of held-out items where the two labels match
    cohen_kappa: float | None  # chance-corrected agreement vs the Council anchor
    min_kappa: float
    ships: bool  # κ measured AND ≥ the gate — the only way a distilled judge goes live


def evaluate_distilled_judge(
    pairs: Sequence[tuple[object, object]],
    *,
    min_kappa: float = ANCHOR_KAPPA_GATE,
) -> DistillAgreement:
    """Score a distilled judge against the Council anchor on a held-out set.

    ``pairs`` = ``(distilled_judge_label, council_anchor_label)`` on the SAME items. Returns the
    raw + chance-corrected agreement and the ship verdict. Empty / κ-undefined ⇒ does NOT ship
    (we never assume agreement)."""
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n if n else 0.0
    kappa = cohen_kappa(pairs)
    ships = kappa is not None and kappa >= min_kappa
    return DistillAgreement(
        n=n,
        raw_agreement=round(po, 6),
        cohen_kappa=round(kappa, 6) if kappa is not None else None,
        min_kappa=min_kappa,
        ships=ships,
    )


__all__ = ["DistillAgreement", "evaluate_distilled_judge"]
