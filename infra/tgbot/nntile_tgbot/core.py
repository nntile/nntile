"""Pure async logic for the bot, independent of aiogram.

Every command's behaviour lives here as a free function taking the
gateway client + chat store + chat/user identifiers explicitly, and
returning either a string (the reply text) or a structured result
(`ModelListReply` for /models, where the renderer also needs the
inline-keyboard buttons). That keeps unit tests free of Telegram
primitives -- the aiogram layer in `handlers.py` is a thin shim.
"""

from __future__ import annotations

from dataclasses import dataclass

from nntile_tgbot.client import GatewayClient, GatewayError, ModelInfo
from nntile_tgbot.state import ChatStore

WELCOME = (
    "Hi! I'm a thin Telegram front-end for an nntile-gateway server.\n\n"
    "Commands:\n"
    "  /models  — list available models with selection buttons\n"
    "  /select <id>  — pick a model by id\n"
    "  /current — show the currently selected model\n"
    "  /reset   — forget conversation history\n"
    "  /fill <text with [MASK]> — fill-mask via the selected BERT/RoBERTa\n"
    "  /help    — show this message\n\n"
    "After selecting a model, just send a message and you'll get a reply."
)


def is_authorized(user_id: int | None, allowed: set[int]) -> bool:
    """True if `user_id` is allowed to talk to the bot.

    An empty `allowed` set means "open to anyone" (still gated by the
    gateway API key). With an allowlist set, missing/unknown users
    are rejected silently (the handler simply returns)."""
    if not allowed:
        return True
    return user_id is not None and user_id in allowed


@dataclass(frozen=True)
class ModelButton:
    """One inline-keyboard button shown for `/models`.

    `callback_data` is a `select:<id>` payload parsed back by
    `parse_select_callback` when the user taps."""

    label: str
    callback_data: str


@dataclass(frozen=True)
class ModelListReply:
    """Rendered response for `/models`: text body + optional buttons.

    Buttons are only included for models in the `ready` state -- a
    loading or errored model is listed in the text but isn't tappable."""

    text: str
    buttons: list[ModelButton]


async def handle_models(client: GatewayClient) -> ModelListReply:
    """Build the response to a `/models` command.

    Lists every model the gateway exposes with family/status decoration.
    Ready models also get an inline-keyboard button so the user can
    tap to select rather than typing `/select <id>`."""
    try:
        models = await client.list_models()
    except GatewayError as exc:
        return ModelListReply(
            text=f"Gateway error listing models: {exc}", buttons=[])
    if not models:
        return ModelListReply(
            text="No models are registered on the gateway yet.", buttons=[])
    lines = ["Available models:"]
    buttons: list[ModelButton] = []
    for m in models:
        suffix = f" ({m.family})" if m.family else ""
        if m.status and m.status != "ready":
            suffix += f" [{m.status}]"
        lines.append(f"  • {m.id}{suffix}")
        if not m.status or m.status == "ready":
            buttons.append(ModelButton(
                label=m.id, callback_data=f"select:{m.id}"))
    if not buttons:
        lines.append("\nNone are in a ready state.")
    else:
        lines.append("\nTap a button or send /select <id>.")
    return ModelListReply(text="\n".join(lines), buttons=buttons)


async def handle_select(
    client: GatewayClient,
    store: ChatStore,
    chat_id: int,
    model_id: str,
) -> str:
    """Validate `model_id` against the gateway and stash it in state.

    Also caches the model's `max_seq_len` so `handle_text` can cap the
    outgoing `max_tokens` and avoid bouncing off the gateway's 400
    for short-seq models like flan-t5-small."""
    model_id = model_id.strip()
    if not model_id:
        return "Usage: /select <model_id>. See /models for the list."
    try:
        models = await client.list_models()
    except GatewayError as exc:
        return f"Gateway error: {exc}"
    known = {m.id: m for m in models}
    if model_id not in known:
        return (
            f"Unknown model {model_id!r}. "
            f"See /models for the list of available ids."
        )
    m = known[model_id]
    if m.status and m.status != "ready":
        return (
            f"Model {model_id!r} is in state "
            f"{m.status!r}, not ready."
        )
    store.set_model(chat_id, model_id, max_seq_len=m.max_seq_len)
    suffix = (
        f" (max_seq_len={m.max_seq_len})"
        if m.max_seq_len is not None else ""
    )
    return f"Selected model: {model_id}{suffix}. History cleared."


def handle_current(store: ChatStore, chat_id: int) -> str:
    """Response to `/current`: which model is currently selected?"""
    state = store.get(chat_id)
    if not state.selected_model:
        return "No model selected. Use /models then /select <id>."
    return f"Current model: {state.selected_model}"


def handle_reset(store: ChatStore, chat_id: int) -> str:
    """Response to `/reset`: drop the message history (keep the model)."""
    store.reset(chat_id)
    return "Conversation history cleared."


async def handle_text(
    client: GatewayClient,
    store: ChatStore,
    chat_id: int,
    text: str,
    max_tokens: int,
) -> str:
    """Default handler for plain user messages -> chat completion.

    Posts the full message history to `/v1/chat/completions` and
    returns the assistant reply. On a `GatewayError` (or any other
    failure mode like the gateway crashing mid-request) the user
    sees a "Gateway error: ..." string and the failed user turn is
    rolled back so a retry doesn't double-count it."""
    text = text.strip()
    if not text:
        return ""
    state = store.get(chat_id)
    if not state.selected_model:
        return (
            "No model selected. Send /models to see options, then "
            "/select <id>."
        )
    store.append(chat_id, "user", text)
    messages = store.messages(chat_id)
    # nntile's max_tokens is a hard cap on the static-allocation seq
    # length, so we must not ask for more than the model's max_seq_len.
    # The gateway will reject otherwise, and short-seq models like
    # flan-t5-small (max_seq_len=16) would never accept the bot's
    # default. Cap here so the user never sees that 400.
    effective_max = max_tokens
    if state.selected_max_seq_len is not None:
        effective_max = min(effective_max, state.selected_max_seq_len)
    try:
        reply = await client.chat_completion(
            model=state.selected_model,
            messages=messages,
            max_tokens=effective_max,
        )
    except GatewayError as exc:
        # Roll back the user turn so retries don't pile up unanswered messages.
        _pop_last(store, chat_id, role="user")
        return f"Gateway error: {exc}"
    except Exception as exc:  # noqa: BLE001 - surface anything to the user
        # Transport errors (httpx.RemoteProtocolError when the gateway
        # crashes mid-request), timeouts, etc. Without this the message
        # handler raises and aiogram silently logs it -- the user just
        # sees no reply.
        _pop_last(store, chat_id, role="user")
        return f"Bot error: {type(exc).__name__}: {exc}"
    store.append(chat_id, "assistant", reply)
    return reply


def _pop_last(store: ChatStore, chat_id: int, role: str) -> None:
    """Best-effort: drop the last message if it matches `role`."""
    msgs = store.messages(chat_id)
    if msgs and msgs[-1]["role"] == role:
        # Rebuild history without the trailing entry.
        store.reset(chat_id)
        for m in msgs[:-1]:
            store.append(chat_id, m["role"], m["content"])


def parse_select_callback(data: str) -> str | None:
    """Return the model id from a 'select:<id>' callback payload, or None."""
    prefix = "select:"
    if not data.startswith(prefix):
        return None
    return data[len(prefix):]


async def handle_fill_mask(
    client: GatewayClient,
    store: ChatStore,
    chat_id: int,
    text: str,
    top_k: int = 5,
) -> str:
    """Run fill-mask on the currently selected model.

    Renders the gateway's structured response into a human-readable
    Telegram message: per-mask, a numbered list of top-k candidates
    with score and the filled-in sequence."""
    state = store.get(chat_id)
    if not state.selected_model:
        return (
            "No model selected. Use /models then /select <id> "
            "(pick a bert/roberta fill_mask model)."
        )
    text = text.strip()
    if not text:
        return "Usage: /fill <text with [MASK]>"
    if "[MASK]" not in text and "<mask>" not in text.lower():
        return (
            "Your text contains no [MASK] token. Example:\n"
            "  /fill Hello I'm a [MASK] model."
        )
    try:
        per_mask = await client.fill_mask(
            model=state.selected_model, text=text, top_k=top_k)
    except GatewayError as exc:
        return f"Gateway error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Bot error: {type(exc).__name__}: {exc}"
    if not per_mask:
        return "No mask predictions returned."
    out_lines: list[str] = []
    for mi, cands in enumerate(per_mask):
        header = (
            f"Mask #{mi + 1}:" if len(per_mask) > 1 else "Predictions:"
        )
        out_lines.append(header)
        for ri, c in enumerate(cands, start=1):
            out_lines.append(
                f"  {ri}. {c.token_str!r}  "
                f"(p={c.score:.4f})  →  {c.sequence}"
            )
    return "\n".join(out_lines)


__all__ = [
    "WELCOME",
    "ModelButton",
    "ModelListReply",
    "ModelInfo",
    "is_authorized",
    "handle_models",
    "handle_select",
    "handle_current",
    "handle_reset",
    "handle_text",
    "handle_fill_mask",
    "parse_select_callback",
]
