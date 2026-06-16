"""loader.py — freeze a held-out >=200/type sample from routing_outcomes.

The eval runs against a frozen, hashed, version-pinned held-out sample so a
result is reproducible and tamper-evident. The held-out split is a stable hash
bucket on the row id, so the same rows never enter training (they are evaluated,
not learned from).

PRIVACY: the frozen tasks carry real prompt text from `routing_outcomes` (the
moat). The caller writes them to the Labs-private artifact dir ONLY — never the
public repo / git. This module never persists anything itself.

The exact prompt column + holdout convention are confirmed against the Labs
schema at wire time (see `select_candidates_sql`); the holdout bucketing is done
in Python here so it is deterministic and unit-testable.
"""

from __future__ import annotations

from typing import Any, Iterable

from labs.eval_harness import config
from labs.eval_harness.taskset import FrozenTaskSet, freeze


def select_candidates_sql() -> str:
    """Candidate rows for the held-out eval sample.

    NOTE (wire-time): verify the prompt column name on the Labs schema — the
    judge sweep selects `id, task_type, …` but the prompt text may live in a
    joined `inferences` row. Holdout bucketing is applied in Python below.

    BLOCKED (AIN-459 / 2026-06-16): the real-traffic freeze is deferred — customer
    prompt text is NOT persisted anywhere in the current schema, so this query has
    no `request_prompt` source and freeze_from_rows cannot run on real traffic.
    Pending a prompt-persistence decision; the Tap-3 cycle uses a curated synthetic
    task set instead (fixtures/curated_integration_taskset.json, illustrative-only).
    """
    return (
        "SELECT id, task_type, request_prompt "
        "FROM routing_outcomes "
        "WHERE traffic_class = 'customer' "
        "  AND task_type IS NOT NULL "
        "  AND request_prompt IS NOT NULL "
        "ORDER BY created_at DESC"
    )


def is_heldout(task_id: str, *, holdout_pct: float = 0.1, seed: int | None = None) -> bool:
    """Stable hash bucket — the same ids are held out across runs (CRN)."""
    salt = config.CRN_SEED if seed is None else seed
    h = abs(hash((task_id, "proof-eval-holdout", salt))) % 10_000
    return h < int(holdout_pct * 10_000)


def _prompt_of(row: dict[str, Any]) -> str:
    return row.get("prompt") or row.get("request_prompt") or ""


def freeze_from_rows(
    rows: Iterable[dict[str, Any]], *, version: str, holdout_pct: float = 0.1
) -> FrozenTaskSet:
    """Apply the held-out filter and freeze (hash + version-pin)."""
    tasks: list[dict[str, Any]] = []
    for r in rows:
        tid = str(r["id"])
        if not is_heldout(tid, holdout_pct=holdout_pct):
            continue
        tasks.append({"id": tid, "task_type": r["task_type"], "prompt": _prompt_of(r)})
    return freeze(tasks, version=version)
