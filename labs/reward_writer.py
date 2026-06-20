"""AIN-542 Step 2b · Tier-A reward writer — re-base reward on verify() (break D2).

The judge workers stopped owning reward (Step 2a). This is the writer that gives
verifiable rows a reward INDEPENDENT of the judge: for succeeded, trainable rows
whose task_type has an INTRINSIC verifier (code / extraction / tool_use — no gold
needed), run ``verify_harness.verify`` over the inference payloads and write
``reward = verify(), reward_source='verify'``. Those rows' reward is now a content
check, not ``(judge_score-1)/4`` → ``corr(reward, judge_score)`` drops below 1.00
on the verifiable subset (the Step 2 acceptance; D2 was the live 1.00 identity).

Pure SQL-strings + mapping (no DB I/O), matching ``labeled_corpus`` — the
Spark/cron runner executes SELECT → ``compute_verify_rewards`` → UPDATE. Runs in
the nightly batch (and hourly as outcomes land, per the v2 cadence). Reasoning
(61% of traffic) has no intrinsic verifier → it stays the Council's job (Tier B),
not forced through verify() here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from labs.task_verifiability import TASK_VERIFIABILITY
from labs.verify_harness import REWARD_SOURCE_VERIFY, VerifySample, verify

# Task types with a no-gold intrinsic verifier — the live-traffic anchor. Derived
# from the map so it can't drift (lockstep test).
INTRINSIC_TASK_TYPES: tuple[str, ...] = tuple(
    sorted(tt for tt, tv in TASK_VERIFIABILITY.items() if tv.intrinsic)
)

_IN_LIST = ", ".join(f"'{tt}'" for tt in INTRINSIC_TASK_TYPES)

# Select succeeded + trainable (Step 0 stamp) verifiable rows not already Tier-A
# sourced, joined to inferences for the payloads. Binds %(limit)s.
VERIFY_REWARD_SELECT_SQL = (
    "SELECT ro.id, ro.task_type, ro.judge_score, "
    "       i.request_payload, i.response_payload "
    "FROM routing_outcomes ro "
    "JOIN inferences i ON i.id = ro.inference_id "
    "WHERE ro.outcome_status = 'succeeded' "
    "  AND NOT ro.exclude_from_training "
    f"  AND ro.task_type IN ({_IN_LIST}) "
    "  AND ro.reward_source IS DISTINCT FROM 'verify' "
    "  AND i.response_payload IS NOT NULL "
    "ORDER BY ro.created_at DESC "
    "LIMIT %(limit)s"
)

# The runner applies this per emitted row. reward_source is set to 'verify'.
VERIFY_REWARD_UPDATE_SQL = (
    "UPDATE public.routing_outcomes "
    "SET reward = $2, reward_source = 'verify' "
    "WHERE id = $1"
)


@dataclass(frozen=True)
class VerifyWrite:
    outcome_id: Any
    reward: float
    reward_source: str
    verifier: str
    detail: str


def compute_verify_rewards(rows: Iterable[dict[str, Any]]) -> list[VerifyWrite]:
    """Map verifiable DB rows -> the reward writes to apply. Rows the verifier
    DEFERS on (no code block, unparseable, etc.) are NOT emitted — they keep
    their existing reward and fall to the Council, never a fake verify pass."""
    out: list[VerifyWrite] = []
    for r in rows:
        result = verify(
            VerifySample(
                task_type=r.get("task_type"),
                request_payload=r.get("request_payload"),
                response_payload=r.get("response_payload"),
            )
        )
        if result.reward is None:
            continue  # deferred → Council; do not overwrite
        out.append(
            VerifyWrite(
                outcome_id=r["id"],
                reward=result.reward,
                reward_source=REWARD_SOURCE_VERIFY,
                verifier=result.verifier,
                detail=result.detail,
            )
        )
    return out
