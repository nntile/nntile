# nntile-gateway

Multi-model, multi-user HTTP gateway in front of nntile inference engines.

- **Admin API** (`/admin/*`) protected by a single admin token from env.
- **OpenAI-compatible API** (`/v1/models`, `/v1/completions`, `/v1/chat/completions`) protected by per-key Bearer auth with an in-process TTL cache.
- Pluggable storage: in-memory by default, SQLite optional.
- One shared `nntile.Context` per process; all registered models live under it.

## Run

```bash
export NNTILE_ADMIN_TOKEN=...
python -m nntile_gateway
```

See `nntile_gateway/config.py` for env vars.

## Test

```bash
pip install -e gateway[test]
pytest gateway/tests
```

Tests use a fake engine and do not require StarPU/CUDA or any real model weights.
