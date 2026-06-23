"""AIN-376 · win/loss matrix + AIQ + blind-judging contract (pure, fixture-verified).

Build-ahead of the Spark batch: locks the aggregation + the self-enhancement-bias blinding so
the only thing the live run adds is real competitor traffic + judge calls."""

from __future__ import annotations

import random

from labs.competitor_winloss import JudgedPair, aggregate_winloss, blind_pair


def test_empty_matrix() -> None:
    assert aggregate_winloss([]) == []


def test_winloss_aiq_and_winrate() -> None:
    pairs = [
        # coding vs openrouter: ainfera wins 2, loses 1
        JudgedPair("coding", "openrouter", 0.9, 0.5, 0.01, 0.02),
        JudgedPair("coding", "openrouter", 0.8, 0.4),
        JudgedPair("coding", "openrouter", 0.3, 0.7),  # loss
        # chat vs martian: a tie
        JudgedPair("chat", "martian", 0.6, 0.6),
    ]
    m = {(c.task_class, c.competitor): c for c in aggregate_winloss(pairs)}

    cod = m[("coding", "openrouter")]
    assert (cod.n, cod.wins, cod.losses, cod.ties) == (3, 2, 1, 0)
    assert cod.win_rate == round(2 / 3, 6)
    assert cod.aiq == round((0.4 + 0.4 - 0.4) / 3, 6)  # mean quality delta
    assert cod.cost_ainfera_mean < cod.cost_competitor_mean  # Pareto inputs carried

    chat = m[("chat", "martian")]
    assert (chat.wins, chat.losses, chat.ties) == (0, 0, 1)
    assert chat.win_rate == 0.0  # all ties ⇒ no decisive games


def test_blind_pair_anonymises_and_is_deblindable() -> None:
    rng = random.Random(0)
    orders = set()
    for _ in range(30):
        blinded, mapping = blind_pair("AINFERA_OUT", "COMP_OUT", rng=rng)
        # de-blinding recovers the true source of each slot
        ai_slot = next(k for k, v in mapping.items() if v == "ainfera")
        comp_slot = next(k for k, v in mapping.items() if v == "competitor")
        assert blinded[ai_slot] == "AINFERA_OUT"
        assert blinded[comp_slot] == "COMP_OUT"
        orders.add(ai_slot)
    assert orders == {"A", "B"}  # ainfera lands in BOTH slots ⇒ genuinely blinded
