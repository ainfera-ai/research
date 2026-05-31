"""delta_logger.py — append weekly-rolling done-and-cheaper-vs-baseline to preprint corpus.

Each daily run appends one row to `preprint/delta-log.csv` (or JSONL) with:
  date,policy_version,overall_done_and_cheaper_pct,delta_vs_baseline_pct,n_rows

Rolling 7-day average is the preprint headline number ("Ainfera Inference
saves X% done-and-cheaper vs the incumbent baseline").

Baseline = `ainfera_static` (the q_prior-only policy from `benchmark/run_benchmark.py`)
on a frozen held-out corpus tagged 2026-05-15 (the AIN-1.0 launch corpus).

Append-only: each row is a fact, never mutated. Rollups are computed by
the preprint render job, not by this logger.

References:
  research/preprint/ — render targets the CSV/JSONL produced here
  ainfera-vault methodology/daily-training-cadence.md §"Delta log"
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeltaLogRow:
    date: str                              # ISO-8601 YYYY-MM-DD (Asia/Jakarta)
    policy_version: str                    # vYYYYMMDD-NNN
    overall_done_and_cheaper_pct: float    # live policy on held-out
    delta_vs_baseline_pct: float           # live - baseline
    n_rows: int
    notes: str = ""

    def to_csv_row(self) -> list[str]:
        return [
            self.date,
            self.policy_version,
            f"{self.overall_done_and_cheaper_pct:.4f}",
            f"{self.delta_vs_baseline_pct:.4f}",
            str(self.n_rows),
            self.notes,
        ]

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def append_csv(row: DeltaLogRow, path: Path | None = None) -> Path:
    """Append one row to the preprint CSV; create with header if absent."""
    p = path or Path(
        os.environ.get(
            "LABS_DELTA_LOG_CSV",
            "preprint/delta-log.csv",
        )
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists()
    with p.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow([
                "date",
                "policy_version",
                "overall_done_and_cheaper_pct",
                "delta_vs_baseline_pct",
                "n_rows",
                "notes",
            ])
        w.writerow(row.to_csv_row())
    log.info("delta_logger appended %s to %s", row.policy_version, p)
    return p


def append_jsonl(row: DeltaLogRow, path: Path | None = None) -> Path:
    """Append one row to the preprint JSONL (parallel format)."""
    p = path or Path(
        os.environ.get(
            "LABS_DELTA_LOG_JSONL",
            "preprint/delta-log.jsonl",
        )
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(row.to_json_line() + "\n")
    return p


def build_row(
    *,
    policy_version: str,
    overall_done_and_cheaper_pct: float,
    delta_vs_baseline_pct: float,
    n_rows: int,
    when: datetime | None = None,
    notes: str = "",
) -> DeltaLogRow:
    when = when or datetime.now(tz=timezone.utc)
    return DeltaLogRow(
        date=when.strftime("%Y-%m-%d"),
        policy_version=policy_version,
        overall_done_and_cheaper_pct=round(overall_done_and_cheaper_pct, 6),
        delta_vs_baseline_pct=round(delta_vs_baseline_pct, 6),
        n_rows=int(n_rows),
        notes=notes,
    )
