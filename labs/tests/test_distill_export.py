"""AIN-561 · held-out distill export — split/shape logic (CRN, drop-unlabeled, binarize)."""

from __future__ import annotations

from typing import Any

from labs.council_pointwise import COUNCIL_PASS_MIN_SCORE
from labs.distill_export import shape_example, split_labeled


def _row(i: int, *, score: float | None = 4.0, prompt: str = "p", response: str = "r") -> dict[str, Any]:
    return {
        "id": f"row-{i}",
        "task_type": "chat",
        "request_prompt": prompt,
        "response_text": response,
        "judge_score": score,
    }


def test_shape_binarizes_at_pass_min() -> None:
    passing = shape_example(_row(1, score=float(COUNCIL_PASS_MIN_SCORE)))
    failing = shape_example(_row(2, score=float(COUNCIL_PASS_MIN_SCORE) - 1))
    assert passing is not None and passing.teacher_pass is True
    assert failing is not None and failing.teacher_pass is False


def test_shape_drops_rows_missing_label_or_text() -> None:
    assert shape_example(_row(1, score=None)) is None  # no teacher score
    assert shape_example(_row(2, prompt="")) is None  # no input text (AIN-459 gap)
    assert shape_example(_row(3, response="")) is None  # no response


def test_split_partitions_and_is_deterministic() -> None:
    rows = [_row(i) for i in range(500)]
    train1, held1 = split_labeled(rows, holdout_pct=0.1)
    train2, held2 = split_labeled(rows, holdout_pct=0.1)
    # Deterministic within the process (CRN): identical split across calls.
    assert [e.item_id for e in train1] == [e.item_id for e in train2]
    assert [e.item_id for e in held1] == [e.item_id for e in held2]
    # Disjoint + total = all shaped rows (none lost, none duplicated).
    train_ids, held_ids = {e.item_id for e in train1}, {e.item_id for e in held1}
    assert train_ids.isdisjoint(held_ids)
    assert len(train_ids) + len(held_ids) == 500
    # Roughly holdout_pct held out (uniform hash bucket; wide tolerance for hash noise).
    assert 0.03 < len(held1) / 500 < 0.20


def test_split_drops_unlabeled_rows() -> None:
    rows = [_row(1), _row(2, score=None), _row(3, prompt="")]
    train, held = split_labeled(rows)
    assert len(train) + len(held) == 1  # only the one fully-labeled row survives
