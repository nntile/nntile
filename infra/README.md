# `infra/` — service packages on top of nntile

Two Python packages that turn a built nntile image into an end-user
inference deployment:

| Package | What it is | Needs GPU |
|---|---|---|
| [`gateway/`](gateway/) | OpenAI-compatible HTTP gateway (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/fill_mask`) over nntile inference engines. Multi-model, per-key auth. | yes |
| [`tgbot/`](tgbot/) | Telegram bot front-end that calls the gateway over HTTP. No model weights, no GPU. | no |

Both are installed into the same conda env when you build the main
nntile Docker image, so a single built image runs either or both —
typically as two containers from one image.

## Build the image once

From the repo root:

```bash
docker build -t nntile:latest .
```

That's it. The Dockerfile already does `pip install -e infra/gateway
infra/tgbot` at the end of the `nntile` stage, so the conda env named
`nntile` has both packages on its PYTHONPATH:

```bash
docker run --rm nntile:latest python -m nntile_gateway --help
docker run --rm nntile:latest python -m nntile_tgbot   --help
```

> **First-build note.** The base image compiles StarPU and nntile from
> source for all CUDA archs (sm_70 through sm_120 by default), which
> takes ~30 minutes on a fast box. Subsequent builds reuse the layer
> cache.

---

## Recipe A — two containers on a user-defined bridge network (recommended)

This is the production-ish layout: each service in its own container,
shared image, separate logs, the bot talks to the gateway by service
name via Docker DNS.

### 1. Create a network and a host-side cache dir for HuggingFace

```bash
docker network create nntile-net
mkdir -p $HOME/.cache/nntile-hf
```

### 2. Start the gateway

```bash
ADMIN_TOKEN=$(openssl rand -hex 32)
echo "$ADMIN_TOKEN" > /tmp/nntile-admin-token

docker run -d --rm --name nntile-gw \
    --gpus all \
    --network nntile-net \
    -p 14000:14000 \
    -e NNTILE_ADMIN_TOKEN=$ADMIN_TOKEN \
    -e NNTILE_GATEWAY_HOST=0.0.0.0 \
    -e NNTILE_GATEWAY_PORT=14000 \
    -e NNTILE_GATEWAY_NCPU=1 \
    -e NNTILE_GATEWAY_NCUDA=1 \
    -e CUDA_VISIBLE_DEVICES=1 \
    -v $HOME/.cache/nntile-hf:/workspace/nntile/cache_hf \
    nntile:latest \
    python -m nntile_gateway

# Wait for /healthz
until curl -sf http://localhost:14000/healthz >/dev/null; do sleep 2; done
```

### 3. Register a few models and issue an API key

The gateway has no persistent state by default (in-memory storage), so
this step has to be redone every time you restart the gateway
container. To survive restarts use the SQLite storage backend (set
`NNTILE_GATEWAY_STORAGE=sqlite` and `NNTILE_GATEWAY_SQLITE_PATH=...`).

> **`docker exec -i …`** is intentional: without `-i` the heredoc
> bytes never reach the container's stdin, Python sees EOF immediately
> and exits silently. `-u` keeps stdout unbuffered so per-model
> "ready" lines arrive as each load finishes rather than at the end.

```bash
docker exec -i nntile-gw python -u - <<'PY'
import urllib.request, json, os, time
ADMIN = open("/tmp/nntile-admin-token").read().strip() if False \
    else os.environ["NNTILE_ADMIN_TOKEN"]
BASE = "http://127.0.0.1:14000"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def post(path, body, timeout=900):
    req = urllib.request.Request(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {ADMIN}",
                 "Content-Type": "application/json"},
        data=json.dumps(body).encode(), method="POST",
    )
    return json.loads(opener.open(req, timeout=timeout).read())

specs = [
  {"id":"gpt2","family":"gpt2","hf_name":"gpt2",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"tinyllama","family":"llama",
   "hf_name":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"gpt-neo-125m","family":"gpt_neo",
   "hf_name":"EleutherAI/gpt-neo-125m",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"pythia-70m","family":"gpt_neox",
   "hf_name":"EleutherAI/pythia-70m",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"bert-fill","family":"bert","hf_name":"bert-base-uncased",
   "task":"fill_mask",
   "max_seq_len":128,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"roberta-fill","family":"roberta","hf_name":"roberta-base",
   "task":"fill_mask",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
  {"id":"flan-t5-small","family":"t5","hf_name":"google/flan-t5-small",
   "max_seq_len":256,"dtype":"fp32","batch_size":1,
   "cache_dir":"cache_hf"},
]
for s in specs:
    t0 = time.time()
    r = post("/admin/models", s)
    print(f"  {s['id']:14}  status={r['status']:6}  {time.time()-t0:.1f}s")

key = post("/admin/keys", {"name":"tgbot"})["key"]
print("api_key:", key)
PY
```

Capture the printed `api_key:` value — pass it into the bot env.

### 4. Start the bot

```bash
API_KEY=nnt_...   # from previous step
BOT_TOKEN=...     # from @BotFather

docker run -d --rm --name nntile-bot \
    --network nntile-net \
    -e NNTILE_TGBOT_TOKEN=$BOT_TOKEN \
    -e NNTILE_TGBOT_GATEWAY_URL=http://nntile-gw:8000 \
    -e NNTILE_TGBOT_API_KEY=$API_KEY \
    -e HTTPS_PROXY="$HTTPS_PROXY" \
    -e HTTP_PROXY="$HTTP_PROXY" \
    nntile:latest \
    python -m nntile_tgbot

docker logs -f nntile-bot
```

The bot writes `Run polling for bot @your_bot_username` once it's
talking to Telegram.

### 5. Stop everything

```bash
docker stop nntile-bot nntile-gw
docker network rm nntile-net
```

---

## Environment variable reference

### Gateway

| Variable | Default | Notes |
|---|---|---|
| `NNTILE_ADMIN_TOKEN` | _required_ | Bearer token for `/admin/*` |
| `NNTILE_GATEWAY_HOST` | `127.0.0.1` | Set to `0.0.0.0` in containers |
| `NNTILE_GATEWAY_PORT` | `12224` | The recipes above explicitly set `8000` for a round number |
| `NNTILE_GATEWAY_NCPU` | `-1` | StarPU CPU workers; `-1` = all |
| `NNTILE_GATEWAY_NCUDA` | `-1` | StarPU CUDA workers; `-1` = all |
| `NNTILE_GATEWAY_STORAGE` | `memory` | Set to `sqlite` to persist models/keys |
| `NNTILE_GATEWAY_SQLITE_PATH` | `gateway.sqlite3` | Used when storage=`sqlite` |

See [`gateway/nntile_gateway/config.py`](gateway/nntile_gateway/config.py)
for the full list (auth cache TTL, etc.).

### Telegram bot

| Variable | Default | Notes |
|---|---|---|
| `NNTILE_TGBOT_TOKEN` | _required_ | Bot token from @BotFather |
| `NNTILE_TGBOT_GATEWAY_URL` | `http://127.0.0.1:8000` | Reachable URL of the gateway |
| `NNTILE_TGBOT_API_KEY` | _required_ | Issued by `/admin/keys` |
| `NNTILE_TGBOT_TELEGRAM_PROXY` | _falls back to `HTTPS_PROXY` / `HTTP_PROXY`_ | `http://...` or `https://...` URL; `https://...` proxies require the TLS-to-proxy behaviour that httpx handles natively |
| `NNTILE_TGBOT_ALLOWED_USERS` | _empty_ | Comma-separated Telegram user IDs; empty = open |
| `NNTILE_TGBOT_HISTORY_TURNS` | `8` | Per-chat message history cap |
| `NNTILE_TGBOT_MAX_TOKENS` | `128` | Auto-capped per model from `/v1/models` |
| `NNTILE_TGBOT_REQUEST_TIMEOUT` | `120` | Per-request timeout in seconds |

See [`tgbot/nntile_tgbot/config.py`](tgbot/nntile_tgbot/config.py).

---

## Why does the bot need an HTTP proxy?

If your host needs an outbound proxy to reach `api.telegram.org` (corporate
network etc.), forward it into the bot container via
`-e HTTPS_PROXY=$HTTPS_PROXY -e HTTP_PROXY=$HTTP_PROXY`. The bot uses
`HttpxSession` (not `AiohttpSession`) so a TLS-wrapped HTTPS proxy URL
(`https://user:pass@proxy:port`) works natively — httpx terminates TLS
to the proxy itself. The gateway client inside the bot always sets
`trust_env=False`, so localhost / Docker-network traffic to the
gateway never goes through the proxy.

---

## Pointers

- Gateway API surface and per-package details:
  [`gateway/README.md`](gateway/README.md)
- Bot commands and architecture:
  [`tgbot/README.md`](tgbot/README.md)
- Live multi-family integration test (opt-in, needs CUDA + transformers):
  `pytest infra/gateway/tests/test_live.py --live`
