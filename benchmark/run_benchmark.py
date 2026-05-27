#!/usr/bin/env python3
"""Routing vs baselines on SYNTHETIC data -> the delta table.

Policies:
  agent_baseline   the agent's own fixed default model (the counterfactual the
                   savings-share is measured against)
  single_best      always the strongest model overall
  cheapest         always the cheapest model (naive)
  round_robin      cycle models
  ainfera_static   q_prior only: cheapest model the PRIOR thinks clears the bar
                   (no learning -> inherits the prior's mistakes)
  ainfera_learned  warm-start from q_prior, then LinUCB refines from outcomes
                   (q_empirical) + exploration floor -> corrects the prior
  oracle           cheapest model clearing the bar at TRUE quality (upper bound)

Key realism: the static catalog prior is IMPERFECT (systematic per-cell error).
The learner's job is to beat the prior using real outcome labels -- the whole
q_empirical thesis. Reward is a 1-5 judge label (simulated from latent + noise).
"""
import json, os, math, random
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "datasets")
COMPLETION_BAR = 3.0
MARGIN = 0.08
EXPLORE_FLOOR = 0.08      # >= 5% exploration floor (methodology/exploration-floor.md)
NOISE = 0.5
PRIOR_ERR = 0.45      # std of systematic prior mis-estimation per (model, task)
ALPHA = 0.25          # LinUCB confidence width
WARM = 2              # pseudo-observations seeding the learner from the prior
SEED = 7

def load():
    with open(os.path.join(DATA, "catalog.json")) as f:
        cat = json.load(f)
    reqs = []
    with open(os.path.join(DATA, "synthetic_outcomes.jsonl")) as f:
        for line in f:
            reqs.append(json.loads(line))
    return cat, reqs

def build_prior(latent, models, tasks, seed=11):
    """An imperfect static catalog estimate of quality (what q_prior would be)."""
    rng = random.Random(seed)
    prior = {}
    for m in models:
        prior[m] = {}
        for t in tasks:
            prior[m][t] = float(np.clip(latent[m][t] + rng.gauss(0, PRIOR_ERR), 1.0, 5.0))
    return prior

def true_quality(latent, model, task, rng):
    return float(np.clip(rng.gauss(latent[model][task], NOISE), 1.0, 5.0))

def all_in_cost(price_per_1k, tokens_k):
    return price_per_1k * tokens_k * (1.0 + MARGIN)

class LinUCB:
    def __init__(self, arms, tasks, alpha=ALPHA):
        self.arms, self.tasks = arms, tasks
        self.d = len(tasks) + 1
        self.alpha = alpha
        self.A = {a: np.identity(self.d) for a in arms}
        self.b = {a: np.zeros(self.d) for a in arms}

    def ctx(self, task):
        v = np.zeros(self.d); v[self.tasks.index(task)] = 1.0; v[-1] = 1.0
        return v

    def estimate(self, arm, x):
        Ainv = np.linalg.inv(self.A[arm])
        theta = Ainv @ self.b[arm]
        return float(theta @ x), self.alpha * math.sqrt(float(x @ Ainv @ x))

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

def run():
    cat, reqs = load()
    models = [m["id"] for m in cat["models"]]
    price = {m["id"]: m["price_per_1k"] for m in cat["models"]}
    latent, tasks = cat["latent_quality"], cat["tasks"]
    prior = build_prior(latent, models, tasks)
    rng = random.Random(SEED)

    avg_q = {m: np.mean(list(latent[m].values())) for m in models}
    strongest = max(models, key=lambda m: avg_q[m])
    cheapest_model = min(models, key=lambda m: price[m])
    agent_default = "model-b"

    learner = LinUCB(models, tasks)
    for m in models:                       # warm-start from the prior
        for t in tasks:
            x = learner.ctx(t)
            for _ in range(WARM):
                learner.update(m, x, prior[m][t])

    rr = 0
    keys = ["agent_baseline","single_best","cheapest","round_robin",
            "ainfera_static","ainfera_learned","oracle"]
    R = {p: {"q": [], "cost": [], "complete": [], "cheaper_and_done": []} for p in keys}

    for r in reqs:
        task, tok, bar = r["task_type"], r["tokens_k"], r["min_quality"]
        base_q = true_quality(latent, agent_default, task, rng)
        base_cost = all_in_cost(price[agent_default], tok)

        def record(p, model):
            q = true_quality(latent, model, task, rng)
            c = all_in_cost(price[model], tok)
            R[p]["q"].append(q); R[p]["cost"].append(c)
            R[p]["complete"].append(1.0 if q >= COMPLETION_BAR else 0.0)
            done_cheaper = (q >= COMPLETION_BAR) and (c <= base_cost) and (q >= base_q - 0.25)
            R[p]["cheaper_and_done"].append(1.0 if done_cheaper else 0.0)
            return q, c

        record("agent_baseline", agent_default)
        record("single_best", strongest)
        record("cheapest", cheapest_model)
        record("round_robin", models[rr % len(models)]); rr += 1

        sclear = [m for m in models if prior[m][task] >= bar]
        record("ainfera_static", min(sclear, key=lambda m: price[m]) if sclear else strongest)

        oclear = [m for m in models if latent[m][task] >= bar]
        record("oracle", min(oclear, key=lambda m: price[m]) if oclear else strongest)

        x = learner.ctx(task)
        if rng.random() < EXPLORE_FLOOR:
            chosen = rng.choice(models)
        else:
            est = {m: learner.estimate(m, x) for m in models}
            kept = [m for m in models if est[m][0] >= bar]   # exploit on mean estimate
            if not kept:
                kept = [max(models, key=lambda m: est[m][0])]
            chosen = min(kept, key=lambda m: price[m])
        q, _ = record("ainfera_learned", chosen)
        learner.update(chosen, x, q)  # judge label (1-5) -> reward

    return R, agent_default

def summarize(R, agent_default):
    n = len(next(iter(R.values()))["q"])
    base_cost = float(np.mean(R["agent_baseline"]["cost"]))
    rows = []
    for p, d in R.items():
        rows.append((p, float(np.mean(d["q"])), 100*float(np.mean(d["complete"])),
                     float(np.mean(d["cost"])),
                     100*(1 - float(np.mean(d["cost"]))/base_cost),
                     100*float(np.mean(d["cheaper_and_done"]))))
    rows.sort(key=lambda r: (-r[5], -r[4]))
    print(f"\nRouting benchmark — {n} synthetic requests "
          f"(baseline = agent's own '{agent_default}')\n")
    hdr = f"{'policy':<16}{'qual':>6}{'done%':>8}{'cost':>8}{'save%':>8}{'done&cheaper%':>16}"
    print(hdr); print("-"*len(hdr))
    for p,q,comp,c,save,cad in rows:
        print(f"{p:<16}{q:>6.2f}{comp:>8.1f}{c:>8.2f}{save:>+8.1f}{cad:>16.1f}")
    print("\nReading it:")
    print("  save%         = mean all-in cost cut vs the agent's own baseline")
    print("  done&cheaper% = tasks completed at <= baseline cost AND ~>= baseline quality")
    print("  ainfera_learned should beat ainfera_static (it corrects prior error)")
    print("  and approach oracle. cheapest/single_best trade away one axis for the other.")

if __name__ == "__main__":
    R, base = run()
    summarize(R, base)
