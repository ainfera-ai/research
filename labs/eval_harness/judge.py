"""judge.py — held-out judge adapter (Gemini 3.1 Pro) + human spot-check queue.

Stage 1 STUB: the adapter holds the judge identity and the interface; it makes
no live call (`label()` raises). Stage 2 (AIN-459) wires the real Gemini 3.1 Pro
call on the Labs tenant and labels each arm's output.

Two invariants this module protects:
  L3  The judge is held out of every arm (enforced in arms.assert_judge_held_out;
      asserted again at construction here).
  The judge PROMPT is the moat and is NOT in this public repo — only the identity
  and the success-threshold mapping live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from labs.eval_harness import arms, config


@dataclass(frozen=True)
class JudgeLabel:
    """One judge label for one (arm, task) output."""

    task_id: str
    arm: str
    score: float        # 1.0 - 5.0
    success: bool        # score >= SUCCESS_SCORE_THRESHOLD
    queued_for_human: bool


@runtime_checkable
class Judge(Protocol):
    model: str

    def label(self, *, task: dict[str, Any], arm: str, output: str) -> JudgeLabel:
        ...


class StubJudge:
    """Inert Gemini 3.1 Pro adapter — Stage 1. Refuses to make a live call."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.JUDGE_MODEL
        # Defence in depth: the judge can never also be an arm (L3).
        arms.assert_judge_held_out(judge_model=self.model)

    def label(self, *, task: dict[str, Any], arm: str, output: str) -> JudgeLabel:  # noqa: ARG002
        raise NotImplementedError(
            "judge.label: live Gemini 3.1 Pro labeling lands in Stage 2 (AIN-459). "
            "Stage 1 ships the held-out adapter interface only."
        )


def success_from_score(score: float, threshold: float | None = None) -> bool:
    """Map a 1-5 judge score to a binary success at the configured threshold."""
    return score >= (config.SUCCESS_SCORE_THRESHOLD if threshold is None else threshold)


def queue_for_human(task_id: str, pct: float | None = None, *, seed: int | None = None) -> bool:
    """Deterministically select ~`pct` of labels for human spot-check (CRN).

    No RNG — a stable hash of the task id keeps the selection reproducible across
    re-runs (same idiom as judge_worker.select_sample).
    """
    pct = config.HUMAN_SPOTCHECK_PCT if pct is None else pct
    if pct <= 0:
        return False
    if pct >= 1:
        return True
    salt = config.CRN_SEED if seed is None else seed
    h = abs(hash((task_id, "proof-eval-spotcheck", salt))) % 10_000
    return h < int(pct * 10_000)
