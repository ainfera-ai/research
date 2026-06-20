"""Judge-free labeled corpus — inferences join + completion reward + the
2026-06-17 signal-degeneracy finding pinned as a regression guard.
"""

from __future__ import annotations

from labs import labeled_corpus as lc
from labs.linucb_refit import fit


def _row(task, cand, status, *, has_text=True):
    return {
        "task_type": task,
        "chosen_candidate": cand,
        "outcome_status": status,
        "cost_actual_usd": 0.01,
        "reward": 0.7,
        "has_text": has_text,
    }


# ── the inferences join (the gap loader.py flagged) ──────────────────────────


def test_sql_joins_inferences_and_is_fleet_origin():
    sql = lc.select_labeled_corpus_sql()
    assert "JOIN inferences" in sql
    assert "ro.inference_id" in sql
    assert "fleet_agent IS NOT NULL" in sql  # fleet-origin → sidesteps AIN-481
    assert "ro.reward IS NOT NULL" in sql


# ── judge-free completion reward (ratified decouple term) ────────────────────


def test_completion_reward_is_judge_free():
    assert lc.completion_reward({"outcome_status": "succeeded"}) == 1.0
    assert lc.completion_reward({"outcome_status": "failed_other"}) == 0.0
    assert lc.completion_reward({}) == 0.0  # unknown → 0, never fabricated


def test_assemble_maps_fields_and_drops_textless_rows():
    rows = [
        _row("chat", "a", "succeeded"),
        _row("chat", "b", "failed_other"),
        _row("chat", "c", "succeeded", has_text=False),  # dropped (no text)
    ]
    corpus = lc.assemble_corpus(rows)
    assert len(corpus) == 2
    assert {c["chosen_candidate"] for c in corpus} == {"a", "b"}
    assert corpus[0] == {"task_type": "chat", "chosen_candidate": "a", "reward": 1.0}


# ── judge-free refit path through fit() ──────────────────────────────────────


def test_fit_uses_judge_free_reward_without_judge_score():
    # Rows carry NO judge_score — the judge-free path must not touch it.
    rows = [
        _row("chat", "a", "succeeded"),
        _row("chat", "a", "succeeded"),
        _row("code", "b", "failed_other"),
    ]
    corpus = lc.assemble_corpus(rows)
    policy = fit(corpus, reward_fn=lambda r: r["reward"], seed=1)
    cells = {(c.task_type, c.candidate): c for c in policy.cells}
    assert cells[("chat", "a")].q_empirical == 1.0  # 2/2 succeeded
    assert cells[("code", "b")].q_empirical == 0.0  # 0/1 succeeded


def test_backward_compatible_judge_mapping_still_works():
    # Default reward_fn = (judge_score-1)/4 — unchanged behaviour.
    rows = [{"task_type": "chat", "chosen_candidate": "a", "judge_score": 5.0}]
    policy = fit(rows, seed=1)
    assert policy.cells[0].q_empirical == 1.0  # (5-1)/4


# ── THE FINDING: all-succeeded completion is a degenerate signal ─────────────


def test_all_succeeded_corpus_is_degenerate_and_visible():
    # The live 2026-06-17 reality: every reward row succeeded. A completion-only
    # refit then yields a FLAT policy (q=1.0 everywhere) — it learns nothing.
    rows = [_row(f"t{i % 3}", f"m{i % 4}", "succeeded") for i in range(120)]
    corpus = lc.assemble_corpus(rows)
    # The guard the runner must honour: ~zero variance → HOLD, not silent "train".
    assert lc.corpus_reward_variance(corpus) == 0.0
    policy = fit(corpus, reward_fn=lambda r: r["reward"], seed=1)
    assert all(c.q_empirical == 1.0 for c in policy.cells)  # flat → no signal


# ── B1 cost-aware reward (ratified) — the cost term carries the signal ────────


def _crow(task, cand, cost, status="succeeded", has_text=True):
    return {
        "task_type": task,
        "chosen_candidate": cand,
        "outcome_status": status,
        "cost_actual_usd": cost,
        "has_text": has_text,
    }


def test_cost_aware_cheaper_candidate_scores_higher():
    rows = [_crow("chat", "cheap", 0.0001), _crow("chat", "pricey", 0.1)]
    out = {c["chosen_candidate"]: c["reward"] for c in lc.cost_aware_corpus(rows)}
    assert out["cheap"] > out["pricey"]
    assert out["cheap"] == 1.0 and out["pricey"] == 0.0  # log-min-max endpoints


def test_cost_aware_is_non_degenerate_when_completion_is_flat():
    # The whole point: all rows succeeded (completion flat 1.0) — completion-only
    # has zero variance, but the cost term gives the cost-aware reward signal.
    rows = [_crow(f"t{i % 2}", f"m{i % 4}", 10 ** -(i % 5 + 1)) for i in range(80)]
    comp = lc.assemble_corpus(rows)  # completion-only
    cost = lc.cost_aware_corpus(rows)  # cost-aware
    assert lc.corpus_reward_variance(comp) == 0.0  # degenerate
    assert lc.corpus_reward_variance(cost) > 0.0  # has signal ✓


def test_cost_aware_failed_completion_zeros_reward():
    # 1{succeeded} gates: a failed row scores 0 regardless of cost.
    rows = [_crow("chat", "a", 0.0001, status="failed_other"), _crow("chat", "b", 0.1)]
    out = {c["chosen_candidate"]: c["reward"] for c in lc.cost_aware_corpus(rows)}
    assert out["a"] == 0.0  # failed → 0 even though cheapest


def test_cost_aware_single_cost_no_divide_by_zero():
    rows = [_crow("chat", "a", 0.01), _crow("chat", "b", 0.01)]
    out = lc.cost_aware_corpus(rows)
    assert all(c["reward"] == 1.0 for c in out)  # hi==lo → norm 0 → completion
