"""AIN-542 Step 2b · Tier-A reward writer."""

from __future__ import annotations

import json

from labs.reward_writer import (
    INTRINSIC_TASK_TYPES,
    VERIFY_REWARD_SELECT_SQL,
    VERIFY_REWARD_UPDATE_SQL,
    compute_verify_rewards,
)


def _resp(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def test_intrinsic_types_are_code_extraction_tool_use() -> None:
    assert INTRINSIC_TASK_TYPES == ("code", "extraction", "tool_use")


def test_select_sql_scopes_to_trainable_verifiable_unsourced() -> None:
    for tt in INTRINSIC_TASK_TYPES:
        assert f"'{tt}'" in VERIFY_REWARD_SELECT_SQL
    assert "NOT ro.exclude_from_training" in VERIFY_REWARD_SELECT_SQL  # Step 0 stamp
    assert "reward_source IS DISTINCT FROM 'verify'" in VERIFY_REWARD_SELECT_SQL
    assert "outcome_status = 'succeeded'" in VERIFY_REWARD_SELECT_SQL


def test_update_sql_sets_verify_source() -> None:
    assert "reward_source = 'verify'" in VERIFY_REWARD_UPDATE_SQL
    assert "WHERE id = $1" in VERIFY_REWARD_UPDATE_SQL


def test_compute_emits_verify_rewards_independent_of_judge() -> None:
    rows = [
        {
            "id": "a",
            "task_type": "code",
            "judge_score": 5,
            "response_payload": _resp("```python\nx=1\n```"),
        },
        {
            "id": "b",
            "task_type": "code",
            "judge_score": 5,
            "response_payload": _resp("no code here"),
        },
        {
            "id": "c",
            "task_type": "extraction",
            "judge_score": 2,
            "response_payload": _resp(json.dumps({"k": 1})),
        },
    ]
    writes = compute_verify_rewards(rows)
    by_id = {w.outcome_id: w for w in writes}
    # 'a' (valid python) and 'c' (valid json) get a verify reward
    assert by_id["a"].reward == 1.0 and by_id["a"].reward_source == "verify"
    assert by_id["c"].reward == 1.0
    # 'b' deferred (no code block) — NOT emitted, keeps existing reward
    assert "b" not in by_id


def test_deferred_rows_are_never_written() -> None:
    rows = [
        {
            "id": "x",
            "task_type": "extraction",
            "judge_score": 4,
            "response_payload": _resp("**not json**"),
        },
    ]
    # '**not json**' is a json_parse_fail → reward 0.0 (a real verdict, emitted)
    writes = compute_verify_rewards(rows)
    assert len(writes) == 1 and writes[0].reward == 0.0
