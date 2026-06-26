#!/usr/bin/env python3
"""Promotion gate (offline replay): would a candidate policy beat the live one?

In production this replays a candidate q_empirical artifact against the live
policy on a held-out labeled slice and returns PROMOTE / HOLD. Here it stands in
the comparison between the static (q_prior) and learned policies as an
illustration of the gate's decision rule: promote only on a measurable win that
does not regress the quality bar.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))
import numpy as np
from run_benchmark import run

MIN_DONE_CHEAPER_GAIN = 1.0   # percentage points required to promote
MAX_QUALITY_REGRESS = 0.05    # absolute quality the candidate may lose

def main():
    R, _ = run()
    def m(p): return (float(np.mean(R[p]["q"])),
                      100*float(np.mean(R[p]["cheaper_and_done"])))
    live_q, live_cad = m("ainfera_static")     # incumbent
    cand_q, cand_cad = m("ainfera_learned")    # candidate
    gain = cand_cad - live_cad
    regress = live_q - cand_q
    decision = "PROMOTE" if (gain >= MIN_DONE_CHEAPER_GAIN and regress <= MAX_QUALITY_REGRESS) else "HOLD"
    print(f"live   (static):  quality={live_q:.2f}  done&cheaper%={live_cad:.1f}")
    print(f"cand (learned):   quality={cand_q:.2f}  done&cheaper%={cand_cad:.1f}")
    print(f"gain={gain:+.1f}pp  quality_regress={regress:+.2f}  ->  {decision}")

if __name__ == "__main__":
    main()
