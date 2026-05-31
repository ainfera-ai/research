"""slack_heartbeat.py — post the L14.2 daily-cadence run summary to #labs.

Webhook URL via env `LABS_SLACK_WEBHOOK_URL` (Doppler-rendered). Posts:
  date · policy_version · decision (PROMOTE/HOLD) · overall delta · halted_reason
  + a 1-line vault commit hint for the orchestrator (`runs/<date>.md`)

Failure mode: if webhook 5xx or network fails, log + return False. The
cron orchestrator still completes the vault commit on its own; the Slack
post is observability, not load-bearing.

References:
  ainfera-vault methodology/daily-training-cadence.md §"Heartbeat"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatPayload:
    date: str
    policy_version: str
    decision: str
    overall_delta_pct: float
    halted_reason: str | None
    sampled_count: int
    cost_usd: float
    vault_path: str

    def to_slack_blocks(self) -> dict[str, Any]:
        emoji = "✅" if self.decision == "PROMOTE" else "🛑"
        return {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} *labs daily — {self.date}*\n"
                            f"version: `{self.policy_version}`\n"
                            f"decision: `{self.decision}`\n"
                            f"delta: `{self.overall_delta_pct:+.4f}pp`\n"
                            + (f"halted: `{self.halted_reason}`\n" if self.halted_reason else "")
                            + f"sampled: {self.sampled_count} · cost: ${self.cost_usd:.2f}\n"
                            f"vault: `{self.vault_path}`"
                        ),
                    },
                }
            ]
        }


def post(payload: HeartbeatPayload, *, webhook_url: str | None = None, timeout: float = 5.0) -> bool:
    """Best-effort POST to Slack. Returns True on 2xx; False otherwise."""
    url = webhook_url or os.environ.get("LABS_SLACK_WEBHOOK_URL")
    if not url:
        log.warning("LABS_SLACK_WEBHOOK_URL not set; skipping heartbeat post")
        return False
    body = json.dumps(payload.to_slack_blocks()).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            ok = 200 <= resp.status < 300
            log.info("heartbeat post → %d (%s)", resp.status, "OK" if ok else "FAIL")
            return ok
    except (URLError, TimeoutError) as e:
        log.warning("heartbeat post failed: %s", e)
        return False
