from __future__ import annotations

import pytest

from nntile_tgbot.state import ChatStore


def test_history_capped():
    s = ChatStore(history_turns=3)
    for i in range(5):
        s.append(1, "user", f"m{i}")
    msgs = s.messages(1)
    assert len(msgs) == 3
    assert [m["content"] for m in msgs] == ["m2", "m3", "m4"]


def test_set_model_clears_history():
    s = ChatStore(history_turns=4)
    s.append(1, "user", "hi")
    s.append(1, "assistant", "hello")
    s.set_model(1, "gpt2")
    assert s.get(1).selected_model == "gpt2"
    assert s.messages(1) == []


def test_reset_keeps_model():
    s = ChatStore(history_turns=4)
    s.set_model(1, "gpt2")
    s.append(1, "user", "hi")
    s.reset(1)
    assert s.get(1).selected_model == "gpt2"
    assert s.messages(1) == []


def test_isolated_chats():
    s = ChatStore(history_turns=4)
    s.set_model(1, "gpt2")
    s.set_model(2, "llama")
    assert s.get(1).selected_model == "gpt2"
    assert s.get(2).selected_model == "llama"


def test_history_turns_must_be_positive():
    with pytest.raises(ValueError):
        ChatStore(history_turns=0)
