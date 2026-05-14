"""
Deployed concurrent load test.

Launches `python -m nntile_gateway` as a real subprocess on a free port
(NOT FastAPI TestClient), registers gpt2 on one CUDA device, issues an
API key, then fires N parallel /v1/completions via httpx.AsyncClient.

Verifies under live conditions:
  - Concurrent requests all return 200 with non-empty text.
  - threading.Lock around engine.generate() actually serializes the
    work: concurrent wall time is within tolerance of sequential wall
    time on the same N. (If the lock were broken on a single CUDA
    device, you would either see crashes or impossibly-fast wall time.)

Opt in with:
    pytest gateway/tests/test_live_concurrent.py --live
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.live


ADMIN_TOKEN = "deploy-test-token"
MODEL_ID = "gpt2-small"
MODEL_SPEC = {
    "id": MODEL_ID,
    "family": "gpt2",
    "hf_name": "gpt2",
    "dtype": "fp32",
    "max_seq_len": 64,
    "batch_size": 1,
}
N_CONCURRENT = 8
COMPLETION_PROMPT = "The quick brown fox"
COMPLETION_MAX_TOKENS = 8


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pick_idle_gpu() -> int | None:
    """Return the index of the GPU with lowest memory usage, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    best_idx: int | None = None
    best_used = float("inf")
    for line in out.strip().splitlines():
        idx_s, used_s = (x.strip() for x in line.split(","))
        idx, used = int(idx_s), int(used_s)
        if used < best_used:
            best_used = used
            best_idx = idx
    return best_idx


def _wait_for_healthz(
    base: str, proc: subprocess.Popen, timeout_s: float = 120.0
) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    with httpx.Client(trust_env=False, timeout=2.0) as client:
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"server process exited early with rc={rc} "
                    f"(last healthz error: {last_err!r})")
            try:
                r = client.get(f"{base}/healthz")
                if r.status_code == 200:
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"server did not become healthy in {timeout_s:.0f}s "
        f"(last error: {last_err!r})")


@pytest.fixture(scope="module")
def deployed_server():
    pytest.importorskip("nntile")
    pytest.importorskip("transformers")

    port = _find_free_port()
    gateway_root = Path(__file__).resolve().parents[1]
    cache_dir = os.environ.get(
        "NNTILE_GATEWAY_LIVE_CACHE",
        str(gateway_root / "cache_hf"))

    env = os.environ.copy()
    env["NNTILE_ADMIN_TOKEN"] = ADMIN_TOKEN
    env["NNTILE_GATEWAY_HOST"] = "127.0.0.1"
    env["NNTILE_GATEWAY_PORT"] = str(port)
    env["NNTILE_GATEWAY_NCPU"] = "1"
    env["NNTILE_GATEWAY_NCUDA"] = "1"
    # Intentionally do NOT restrict CUDA_VISIBLE_DEVICES: StarPU's cached
    # bus performance model was built with all GPUs visible, and shrinking
    # the visible set forces a multi-minute recalibration on first start.
    # Instead, pin StarPU's single CUDA worker to an idle GPU via
    # STARPU_WORKERS_CUDAID so we don't fight other users on a shared box.
    gpu = _pick_idle_gpu()
    if gpu is not None:
        env["STARPU_WORKERS_CUDAID"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(gateway_root),
        env.get("PYTHONPATH", ""),
    ]))

    log_fd, log_path_str = tempfile.mkstemp(
        prefix="nntile_gateway_", suffix=".log")
    os.close(log_fd)
    log_path = Path(log_path_str)
    log_file = log_path.open("w")
    print(
        f"\n[deploy] pinned to GPU={env.get('STARPU_WORKERS_CUDAID')} "
        f"port={port} log={log_path}",
        flush=True,
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "nntile_gateway"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            _wait_for_healthz(base, proc)
        except RuntimeError:
            log_file.flush()
            tail = log_path.read_text()[-4000:]
            raise RuntimeError(f"server startup failed; log tail:\n{tail}")

        admin = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        with httpx.Client(trust_env=False) as client:
            r = client.post(
                f"{base}/admin/models",
                headers=admin,
                json={**MODEL_SPEC, "cache_dir": cache_dir},
                timeout=600.0,
            )
            assert r.status_code == 200, (
                f"model register failed: {r.status_code} {r.text}")
            assert r.json()["status"] == "ready"

            r = client.post(
                f"{base}/admin/keys",
                headers=admin,
                json={"name": "loadtest"},
                timeout=10.0,
            )
            assert r.status_code == 200, r.text
            api_key = r.json()["key"]

        yield base, api_key
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()


async def _post_completion(
    client: httpx.AsyncClient, base: str, api_key: str
) -> httpx.Response:
    return await client.post(
        f"{base}/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL_ID,
            "prompt": COMPLETION_PROMPT,
            "max_tokens": COMPLETION_MAX_TOKENS,
        },
        timeout=120.0,
    )


async def _run_workload(base: str, api_key: str):
    async with httpx.AsyncClient(trust_env=False) as client:
        # Warmup so JIT/perfmodel costs don't pollute the timed runs.
        warm = await _post_completion(client, base, api_key)
        assert warm.status_code == 200, warm.text

        t0 = time.perf_counter()
        seq_responses = []
        for _ in range(N_CONCURRENT):
            r = await _post_completion(client, base, api_key)
            seq_responses.append(r)
        seq_wall = time.perf_counter() - t0

        t0 = time.perf_counter()
        conc_responses = await asyncio.gather(*[
            _post_completion(client, base, api_key)
            for _ in range(N_CONCURRENT)
        ])
        conc_wall = time.perf_counter() - t0
    return seq_wall, conc_wall, seq_responses, conc_responses


def test_concurrent_completions_serialize_correctly(
    deployed_server, capsys
):
    base, api_key = deployed_server

    seq_wall, conc_wall, seq_responses, conc_responses = asyncio.run(
        _run_workload(base, api_key))

    for r in seq_responses + conc_responses:
        assert r.status_code == 200, f"{r.status_code} {r.text}"
        body = r.json()
        assert body["model"] == MODEL_ID
        assert body["choices"][0]["text"] != ""
        assert body["usage"]["completion_tokens"] > 0

    seq_avg = seq_wall / N_CONCURRENT
    conc_avg = conc_wall / N_CONCURRENT
    ratio = conc_wall / seq_wall if seq_wall > 0 else float("inf")
    speedup = seq_wall / conc_wall if conc_wall > 0 else float("inf")

    with capsys.disabled():
        print()
        print(f"  n={N_CONCURRENT} requests, max_tokens={COMPLETION_MAX_TOKENS}")
        print(f"  sequential wall: {seq_wall:.3f}s ({seq_avg*1000:.1f} ms/req)")
        print(f"  concurrent wall: {conc_wall:.3f}s ({conc_avg*1000:.1f} ms/req)")
        print(f"  ratio (conc/seq): {ratio:.3f}")
        print(f"  speedup (seq/conc): {speedup:.3f}x")

    # threading.Lock around engine.generate must serialize: concurrent
    # wall should be close to sequential wall on a single GPU. Allow up
    # to 1.5x speedup to absorb scheduling/IO overlap, but no more.
    assert speedup < 1.5, (
        f"concurrent ran {speedup:.2f}x faster than sequential "
        f"(seq={seq_wall:.2f}s, conc={conc_wall:.2f}s) -- "
        "the generate lock may not be holding"
    )


def test_unauthenticated_request_rejected_when_deployed(deployed_server):
    base, _ = deployed_server
    with httpx.Client(trust_env=False, timeout=10.0) as client:
        r = client.get(f"{base}/v1/models")
        assert r.status_code == 401
        r = client.get(
            f"{base}/v1/models",
            headers={"Authorization": "Bearer nope"},
        )
        assert r.status_code == 401


def test_admin_endpoint_rejects_user_key(deployed_server):
    base, api_key = deployed_server
    with httpx.Client(trust_env=False, timeout=10.0) as client:
        r = client.get(
            f"{base}/admin/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    assert r.status_code == 403
