"""Per-chat conversation state, kept in memory.

This is intentionally a single in-process dict. A bot restart clears state,
which is fine for a developer-facing tool. Persistence can be added later
(SQLite or aiogram's built-in FSM storage) without changing the handler API.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ChatState:
    selected_model: str | None = None
    # max_seq_len of the selected model, used by handlers to cap
    # outgoing max_tokens so we don't bounce off the gateway's
    # validation 400. Populated on /select when the gateway returns it
    # via the /v1/models extension field; None means "unknown, don't cap".
    selected_max_seq_len: int | None = None
    history: deque[dict[str, str]] = field(default_factory=deque)


class ChatStore:
    """Maps chat_id -> ChatState.

    Threadsafe; aiogram dispatches handlers concurrently."""

    def __init__(self, history_turns: int) -> None:
        if history_turns < 1:
            raise ValueError("history_turns must be >= 1")
        self._history_turns = history_turns
        self._states: dict[int, ChatState] = {}
        self._lock = Lock()

    def get(self, chat_id: int) -> ChatState:
        with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                state = ChatState(history=deque(maxlen=self._history_turns))
                self._states[chat_id] = state
            return state

    def set_model(
        self, chat_id: int, model_id: str,
        max_seq_len: int | None = None,
    ) -> None:
        with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                state = ChatState(history=deque(maxlen=self._history_turns))
                self._states[chat_id] = state
            state.selected_model = model_id
            state.selected_max_seq_len = max_seq_len
            # Switching model invalidates prior assistant context.
            state.history.clear()

    def append(self, chat_id: int, role: str, content: str) -> None:
        with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                state = ChatState(history=deque(maxlen=self._history_turns))
                self._states[chat_id] = state
            state.history.append({"role": role, "content": content})

    def messages(self, chat_id: int) -> list[dict[str, str]]:
        with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                return []
            return list(state.history)

    def reset(self, chat_id: int) -> None:
        with self._lock:
            state = self._states.get(chat_id)
            if state is not None:
                state.history.clear()
