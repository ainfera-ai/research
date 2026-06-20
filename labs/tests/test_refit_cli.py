"""Activation wiring — refit_policy composition + the `python3 -m labs.linucb_refit`
runner that env-gates D1 shrinkage + D7 Thompson (AIN-542)."""

from __future__ import annotations

import json

from labs.linucb_refit import _main, fit, refit_policy

REWARD = lambda r: r["reward"]  # noqa: E731


def _rows() -> list[dict]:
    return [
        {"task_type": "t", "chosen_candidate": "good", "reward": 0.9} for _ in range(40)
    ] + [
        {"task_type": "t", "chosen_candidate": "newbad", "reward": 0.1}
        for _ in range(2)
    ]


# ── refit_policy composition ───────────────────────────────────────────────


def test_default_is_byte_identical_to_fit() -> None:
    rows = _rows()
    assert (
        refit_policy(rows, seed=42, reward_fn=REWARD).to_json()
        == fit(rows, seed=42, reward_fn=REWARD).to_json()
    )


def test_shrinkage_only() -> None:
    pol = refit_policy(
        _rows(),
        seed=42,
        reward_fn=REWARD,
        q_priors={"good": 0.9, "newbad": 0.9},
        prior_strength=20,
    )
    j = pol.to_json()
    assert "q_prior" in j and "alloc_weight" not in j


def test_thompson_only() -> None:
    pol = refit_policy(
        _rows(), seed=42, reward_fn=REWARD, thompson=True, min_samples=30
    )
    by = {c.candidate: c.alloc_weight for c in pol.cells}
    assert by["newbad"] >= 0.05  # min-sample floor rescues the young cell
    assert "alloc_weight" in pol.to_json()


def test_both_compose() -> None:
    pol = refit_policy(
        _rows(),
        seed=42,
        reward_fn=REWARD,
        q_priors={"good": 0.9, "newbad": 0.9},
        prior_strength=20,
        thompson=True,
    )
    j = pol.to_json()
    assert "q_prior" in j and "alloc_weight" in j


# ── CLI runner (env-gated) ─────────────────────────────────────────────────


def test_cli_default_off_is_v0(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LABS_PRIOR_STRENGTH", raising=False)
    monkeypatch.delenv("LABS_THOMPSON", raising=False)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(_rows()))
    out = tmp_path / "policy.json"
    rc = _main(["--corpus", str(corpus), "--output", str(out), "--date", "2026-06-20"])
    assert rc == 0
    j = out.read_text()
    assert "q_prior" not in j and "alloc_weight" not in j  # v0 candidate
    assert '"version": "v20260620-001"' in j


def test_cli_activated_via_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LABS_PRIOR_STRENGTH", "20")
    monkeypatch.setenv("LABS_THOMPSON", "true")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(_rows()))
    priors = tmp_path / "priors.json"
    priors.write_text(json.dumps({"good": 0.9, "newbad": 0.9}))
    out = tmp_path / "policy.json"
    rc = _main(
        [
            "--corpus",
            str(corpus),
            "--output",
            str(out),
            "--priors",
            str(priors),
            "--date",
            "2026-06-20",
        ]
    )
    assert rc == 0
    j = out.read_text()
    assert "q_prior" in j and "alloc_weight" in j
