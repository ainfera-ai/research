"""AIN-561 · held-out distill dataset export — teacher+Council labeled examples.

Splits labeled `routing_outcomes` rows into train / held-out via the SAME stable hash
bucket the eval harness uses (`labs.eval_harness.loader.is_heldout`), so a row held out for
distill EVAL is never trained on (CRN). The student judge is distilled on ``train``; its
verdicts are scored against the teacher/Council labels on ``heldout`` by the existing ship
gate (`labs/distill_judge_gate.py`, κ ≥ 0.60 vs the Council anchor — agreement is the gate,
never assumed).

DATA AVAILABILITY (AIN-459): real customer prompt text is NOT persisted in
`routing_outcomes` (privacy), so a real-traffic export is BLOCKED pending the
prompt-persistence decision — the same gap that defers the eval freeze in
`eval_harness/loader.py`. The split + shaping logic here is exercised on fixtures / the
curated synthetic task set and is ready to run on real rows the moment the input text is
available. PRIVACY: never persist prompt text to the repo / git; a caller writes shaped
examples to the Labs-private artifact dir only — this module persists nothing itself.

Screening-only — the distilled judge does NOT enter the moat or govern routing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from labs.council_pointwise import COUNCIL_PASS_MIN_SCORE
from labs.eval_harness.loader import is_heldout


@dataclass(frozen=True)
class DistillExample:
    """One labeled (input → teacher judgment) example for distillation."""

    item_id: str
    task_type: str | None
    prompt: str
    response: str
    teacher_score: float  # Opus pointwise 1–5
    teacher_pass: bool  # binarized at COUNCIL_PASS_MIN_SCORE — the anchor-κ axis


def shape_example(row: dict[str, Any]) -> DistillExample | None:
    """Shape one labeled row into a `DistillExample`, or ``None`` if it lacks a teacher
    score or the input text (the AIN-459 real-traffic gap) — never a fabricated label or
    input. Binarizes the teacher's 1–5 score at ``COUNCIL_PASS_MIN_SCORE`` so the student's
    target matches the axis the κ gate scores on."""
    score = row.get("judge_score")
    prompt = row.get("request_prompt") or row.get("prompt") or ""
    response = row.get("response_text") or row.get("response") or ""
    if score is None or not prompt or not response:
        return None
    s = float(score)
    return DistillExample(
        item_id=str(row["id"]),
        task_type=row.get("task_type"),
        prompt=prompt,
        response=response,
        teacher_score=s,
        teacher_pass=s >= COUNCIL_PASS_MIN_SCORE,
    )


def split_labeled(
    rows: Iterable[dict[str, Any]], *, holdout_pct: float = 0.1
) -> tuple[list[DistillExample], list[DistillExample]]:
    """Return ``(train, heldout)`` on the stable `is_heldout` bucket.

    Rows missing a teacher score or input text are dropped (honest — no fabricated
    label/input). Deterministic: the same id always lands on the same side (CRN), so a
    held-out row is never silently trained on.
    """
    train: list[DistillExample] = []
    heldout: list[DistillExample] = []
    for r in rows:
        ex = shape_example(r)
        if ex is None:
            continue
        if is_heldout(ex.item_id, holdout_pct=holdout_pct):
            heldout.append(ex)
        else:
            train.append(ex)
    return train, heldout
