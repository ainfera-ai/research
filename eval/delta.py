#!/usr/bin/env python3
"""Compute the routing-delta table and write it into the preprint as Markdown.

This is the script behind the preprint headline number. It runs the benchmark on
synthetic data and emits preprint/results.md. Swap the loader for a private,
held-out labeled corpus to produce the REAL number (not in this repo).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))
import numpy as np
from run_benchmark import run, summarize  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "preprint", "results.md")

def main():
    R, base = run()
    base_cost = float(np.mean(R["agent_baseline"]["cost"]))
    lines = ["# Results (synthetic)\n",
             f"Baseline = agent's own default (`{base}`). "
             "`save%` is cost cut vs that baseline; `done&cheaper%` is the share of "
             "tasks completed at <= baseline cost and ~>= baseline quality.\n",
             "| policy | quality | done% | cost | save% | done&cheaper% |",
             "|---|---|---|---|---|---|"]
    rows = []
    for p, d in R.items():
        rows.append((p, float(np.mean(d["q"])), 100*float(np.mean(d["complete"])),
                     float(np.mean(d["cost"])),
                     100*(1-float(np.mean(d["cost"]))/base_cost),
                     100*float(np.mean(d["cheaper_and_done"]))))
    for p,q,comp,c,save,cad in sorted(rows, key=lambda r:(-r[5],-r[4])):
        lines.append(f"| {p} | {q:.2f} | {comp:.1f} | {c:.2f} | {save:+.1f} | {cad:.1f} |")
    open(OUT,"w").write("\n".join(lines)+"\n")
    print(f"wrote {OUT}")
    summarize(R, base)

if __name__ == "__main__":
    main()
