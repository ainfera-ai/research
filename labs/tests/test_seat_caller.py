"""AIN-542 Step 4 · live seat caller (parse, family map, gateway wrapper)."""

from __future__ import annotations

from labs.council_seats import COUNCIL_SEATS
from labs.seat_caller import family_of_slug, gateway_seat_caller, parse_pick


def test_parse_pick() -> None:
    assert parse_pick("FIRST") == "first"
    assert parse_pick("SECOND") == "second"
    assert parse_pick("TIE") == "tie"
    assert parse_pick("The first response is better.") == "first"
    assert parse_pick("They are equal") == "tie"
    assert parse_pick("") == "tie"
    assert (
        parse_pick("uhh both have merit and FIRST... SECOND") == "tie"
    )  # ambiguous → tie


def test_family_of_slug_real_roster() -> None:
    assert family_of_slug("claude-opus-4-7") == "anthropic"
    assert family_of_slug("gpt-5-5") == "openai"
    assert family_of_slug("gemini-3-1-pro") == "google"
    assert family_of_slug("grok-4") == "xai"
    assert family_of_slug("llama-4-405b-together") == "meta"
    assert family_of_slug("deepseek-v4-pro-deepinfra") == "deepseek"
    assert family_of_slug("minimax-m3-novita") == "minimax"
    assert family_of_slug("qwen-3-5-397b-novita") == "alibaba"
    assert family_of_slug(None) == "unknown"
    assert family_of_slug("some-mystery-model") == "unknown"


class _FakeResp:
    def __init__(self, text):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": text})()})()
        ]


class _FakeClient:
    def __init__(self, text=None, raise_exc=False):
        self._text, self._raise = text, raise_exc
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        if self._raise:
            raise RuntimeError("gateway 500")
        return _FakeResp(self._text)


def test_gateway_seat_caller_parses_and_abstains() -> None:
    seat = COUNCIL_SEATS[0]
    ok = gateway_seat_caller(_FakeClient("SECOND"))
    assert ok(seat, "a", "b") == "second"
    # a flaky seat abstains, never crashes
    flaky = gateway_seat_caller(_FakeClient(raise_exc=True))
    assert flaky(seat, "a", "b") == "tie"
