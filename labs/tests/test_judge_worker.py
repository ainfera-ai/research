"""Tests for labs.judge_worker. Deterministic (no DB, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.judge_worker import select_sample, select_unlabeled_sql


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def unlabeled_rows():
    return json.loads((FIXTURES / "synthetic_unlabeled.json").read_text())


def test_select_sample_deterministic(unlabeled_rows):
    """Same inputs → same selected rows (CRN invariant)."""
    sel1 = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.20,
        cell_label_counts={},
    )
    sel2 = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.20,
        cell_label_counts={},
    )
    assert sel1 == sel2, "select_sample must be deterministic"


def test_cold_start_picks_more_when_cell_underlabeled(unlabeled_rows):
    """Cells with <10 labels use cold_start_pct (higher target)."""
    cold_selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.50,  # high cold-start
        cell_label_counts={},  # all cells in cold-start regime
    )
    warm_selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.50,
        # mark all cells fully warm — should use target_pct (0.05) only
        cell_label_counts={
            ("code", "claude-opus-4-7"): 100,
            ("code", "claude-sonnet-4-6"): 100,
            ("chat", "gpt-5-5"): 100,
            ("chat", "claude-sonnet-4-6"): 100,
            ("research", "deepseek-r1-32b"): 100,
            ("research", "claude-opus-4-7"): 100,
        },
    )
    assert len(cold_selected) >= len(warm_selected), (
        "cold-start should pick at least as many rows as warm at the same seed"
    )


def test_sample_max_caps_count(unlabeled_rows):
    """sample_max=2 returns at most 2 rows even with cold_start=100%."""
    selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=1.0,
        sample_max=2,
        cold_start_pct=1.0,
        cell_label_counts={},
    )
    assert len(selected) <= 2


def test_empty_input_returns_empty():
    assert (
        select_sample(
            unlabeled_rows=[],
            sample_target_pct=0.05,
            sample_max=100,
            cold_start_pct=0.20,
            cell_label_counts={},
        )
        == []
    )


def test_run_one_cycle_skeleton_raises():
    """W6 skeleton — real DB + Opus integration lands in AIN-290 follow-up."""
    from labs.judge_worker import run_one_cycle

    with pytest.raises(NotImplementedError):
        run_one_cycle()


# --- AIN-298 W7 cadence guard ---------------------------------------------


def test_query_guards_null_inference_id():
    """The judge sweep MUST filter out NULL inference_id rows.

    Without this guard, the worker would try to score reject-path outcomes
    (no_candidate_clears_floor) and pre-dispatch-failure outcomes (AIN-300
    W1 'failed_pre_dispatch') against an inference row that doesn't exist —
    the cron would burn $ on Opus calls that can't possibly produce a
    valid judge_score.
    """
    sql = select_unlabeled_sql()
    assert "inference_id IS NOT NULL" in sql, (
        "AIN-298 W7 guard missing — `inference_id IS NOT NULL` must remain "
        "in the judge sweep query so reject-path + pre-dispatch-failure rows "
        "never enter the queue. See ainfera-vault methodology/daily-training-"
        "cadence.md §judge-sweep + AIN-300 W1 decision_rule_override."
    )


def test_query_guards_unlabeled_only():
    """Belt-and-suspenders — the judge sweep also filters judge_status."""
    sql = select_unlabeled_sql()
    assert "judge_status = 'unlabeled'" in sql, (
        "judge sweep must only pick rows that have never been labeled"
    )


def test_query_guards_succeeded_outcome_only():
    """W1 carry-over — only succeeded inferences are score-able. failed_*
    outcomes have an inference row (post-W1 linkage) but no response
    payload to score.
    """
    sql = select_unlabeled_sql()
    assert "outcome_status = 'succeeded'" in sql, (
        "judge sweep must only pick outcomes that completed successfully"
    )
