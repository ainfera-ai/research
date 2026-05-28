"""Ainfera Labs — daily training cadence (L14.2).

Anchored at 03:00 WIB nightly on the DGX Spark. Workflow:

  03:00  judge_worker.py      Opus 4.7 labels last-24h sample (≥5%, ≤100/run)
  03:30  linucb_refit.py      bandit refit on rolling 30d labeled corpus
  04:00  replay_gate.py       CRN harness; PROMOTE / HOLD
  04:15  (policy publish)     api/admin/policy/publish (W6-B)
  04:30  delta_logger.py      done-and-cheaper-vs-baseline appended to preprint corpus
  04:45  slack_heartbeat.py   run summary to #labs + vault commit

Promote criterion (Discipline #12 — moat):
  - replay_delta ≥ +0.5% done-and-cheaper-vs-baseline
  - exploration floor ≥5% preserved
  - no cell regresses ≤-2%
  - sample size ≥30 rows/cell

Cost envelope: ≤100 judge calls/day (~$10-15 at Opus 4.7 rates). Cumulative
spend halted at $15/day with alert.

References:
  ainfera-vault methodology/daily-training-cadence.md
  ainfera-vault decisions/locks-2026-05-28-l14.md §L14.2
"""

from __future__ import annotations

__version__ = "0.1.0"
__l14_cadence__ = "daily-03:00-WIB"
