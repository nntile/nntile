# nntile-tgbot

Telegram bot front-end for the [nntile-gateway](../gateway) HTTP API. The bot
calls the gateway over HTTP only — it doesn't import nntile or load any model
weights itself.

## Layout

- `nntile_tgbot/core.py` — pure async logic (no aiogram, no Telegram types),
  easy to unit test.
- `nntile_tgbot/handlers.py` — aiogram v3 `Router` that wraps the core
  functions with Telegram message/callback handling.
- `nntile_tgbot/client.py` — async httpx wrapper around `/v1/models` and
  `/v1/chat/completions`.
- `nntile_tgbot/state.py` — per-chat in-memory store (selected model + a
  bounded message history).
- `nntile_tgbot/config.py` — env-driven config (`BotConfig`).

## Run

```bash
pip install -e infra/tgbot

export NNTILE_TGBOT_TOKEN=...                    # from @BotFather
export NNTILE_TGBOT_GATEWAY_URL=http://127.0.0.1:8000
export NNTILE_TGBOT_API_KEY=nnt_...              # a key issued by /admin/keys
# optional:
# export NNTILE_TGBOT_ALLOWED_USERS=123456,234567
# export NNTILE_TGBOT_HISTORY_TURNS=8
# export NNTILE_TGBOT_MAX_TOKENS=128

python -m nntile_tgbot
```

## Commands

- `/start`, `/help` — usage info
- `/models` — list models on the gateway with inline selection buttons
- `/select <id>` — pick a model by id (text fallback if buttons aren't handy)
- `/current` — show the currently selected model
- `/reset` — forget conversation history (model stays selected)

Any other message is sent to `/v1/chat/completions` for the selected model
with a short rolling history.

## Test

```bash
pip install -e infra/tgbot[test]
pytest infra/tgbot/tests
```

Tests use an in-process `httpx.MockTransport` — no real Telegram API and no
running gateway required.
