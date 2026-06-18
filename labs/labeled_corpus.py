"""labeled_corpus.py — assemble the judge-free labeled corpus for the LinUCB refit.

Closes the gap `eval_harness/loader.py` flagged: the prompt/completion text is NOT
in `routing_outcomes` (the AIN-459/481 note) — it lives in **`inferences`**, joinable
by `inference_id`. This module provides the join + the judge-free reward so the
refit (`linucb_refit.fit`) can run on real fleet-origin rows.

Scope (per counsel audit vault://sync/2026-06-17-deliver-all-report-audit.md):
- **Fleet-origin only** (`fleet_agent IS NOT NULL`) — own dogfood text, which
  **sidesteps AIN-481** (no customer-privacy gate on internal rows).
- **Judge-free reward** = the *completion* term of the ratified decouple
  (`decisions/2026-06-13-reward-decouple.md`): `completion` from `outcome_status`
  (succeeded→1.0, else→0.0). The quality/judge term is parked under the κ-HOLD;
  completion-based learning may proceed under the HOLD.

⚠️ KNOWN SIGNAL CAVEAT (measured 2026-06-17 on the live corpus): every reward row
to date has `outcome_status='succeeded'` → the completion reward is **uniformly
1.0 (zero variance)**, so a refit on completion alone learns nothing (flat policy;
the replay-gate correctly HOLDs). A non-degenerate judge-free signal needs ONE of:
the κ-HOLD clearing (quality term), a **cost-aware "done-and-cheaper" reward**
(cost varies → has signal; a Námo reward-design decision — PROPOSED, not coded here
to avoid pre-empting the moat call), or actual completion failures once the fleet
runs real, sometimes-failing work. `corpus_reward_variance()` makes the
degeneracy observable so the pipeline never silently "trains" on a flat signal.

This module is pure + SQL-string only (no DB I/O), matching `eval_harness/loader`.
The Spark/cron runner executes the SQL and feeds rows to `assemble_corpus`.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any, Callable

RewardFn = Callable[[dict[str, Any]], float]

# Fleet-origin trainable corpus, joined to `inferences` for the prompt/completion
# text (presence gate) the refit's provenance + AIN-481 scope rely on. Parameterized
# by :days (rolling window). The runner binds %(days)s.
LABELED_CORPUS_SQL = (
    "SELECT ro.id, ro.task_type, ro.chosen_model_slug AS chosen_candidate, "
    "       ro.outcome_status, ro.cost_actual_usd, ro.reward, "
    "       (i.request_payload IS NOT NULL AND i.response_payload IS NOT NULL) AS has_text "
    "FROM routing_outcomes ro "
    "JOIN inferences i ON i.id = ro.inference_id "
    "WHERE ro.reward IS NOT NULL "
    "  AND ro.fleet_agent IS NOT NULL "
    "  AND ro.task_type IS NOT NULL AND ro.chosen_model_slug IS NOT NULL "
    "  AND ro.created_at >= now() - (%(days)s || ' days')::interval "
    "ORDER BY ro.created_at DESC"
)


def select_labeled_corpus_sql() -> str:
    """The fleet-origin labeled-corpus query (routing_outcomes ⋈ inferences)."""
    return LABELED_CORPUS_SQL


def completion_reward(row: dict[str, Any]) -> float:
    """Judge-free completion reward: 1.0 iff the inference succeeded, else 0.0
    (ratified decouple completion term). None/unknown status → 0.0."""
    return 1.0 if row.get("outcome_status") == "succeeded" else 0.0


def assemble_corpus(
    rows: Iterable[dict[str, Any]],
    *,
    reward_fn: RewardFn = completion_reward,
    require_text: bool = True,
) -> list[dict[str, Any]]:
    """Map DB rows → the ``{task_type, chosen_candidate, reward}`` records
    ``linucb_refit.fit`` consumes, computing the judge-free reward. With
    ``require_text`` (default), drop rows whose inferences row lacked both
    payloads — keeping the AIN-481-clear, fully-texted trainable set."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if require_text and not r.get("has_text"):
            continue
        out.append(
            {
                "task_type": r["task_type"],
                "chosen_candidate": r["chosen_candidate"],
                "reward": reward_fn(r),
            }
        )
    return out


def corpus_reward_variance(corpus: list[dict[str, Any]]) -> float:
    """Population variance of the assembled reward. ~0 means a degenerate signal
    (e.g. all-succeeded completion) — the refit would learn nothing. The runner
    should treat near-zero variance as a HOLD, not a silent flat 'training'."""
    rewards = [c["reward"] for c in corpus]
    if len(rewards) < 2:
        return 0.0
    return statistics.pvariance(rewards)
