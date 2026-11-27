# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file logger/test_server.py
# LOG_DIR clearing policy of the TensorBoard logger server.
#
# Run explicitly (needs tensorflow + tensorboard on PATH):
#     pytest logger/test_server.py

import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("tensorboard") is None,
    reason="tensorboard is not installed",
)


def _run_server(log_dir: Path, env_extra: dict) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(env_extra)
    # server.py uses plain print(); without this its startup lines sit
    # in the pipe buffer and the waiter below would block on readline.
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _wait_started(proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "server did not start in time"
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            raise AssertionError(
                f"server exited early: {proc.returncode}"
            )
        if "Server has been started" in line:
            return


def _stop(proc: subprocess.Popen) -> None:
    # Kill the whole group: server.py also spawns a tensorboard child.
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=15)


def test_server_keeps_prior_logs_by_default(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    sentinel = log_dir / "prior-run-events"
    sentinel.write_text("keep me")

    proc = _run_server(
        log_dir, {"LOG_DIR": str(log_dir), "SERVER_PORT": "0"}
    )
    try:
        _wait_started(proc)
        assert sentinel.read_text() == "keep me"
        assert log_dir.is_dir()
    finally:
        _stop(proc)


def test_server_clears_logs_only_when_opted_in(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    sentinel = log_dir / "prior-run-events"
    sentinel.write_text("wipe me")

    proc = _run_server(
        log_dir,
        {
            "LOG_DIR": str(log_dir),
            "SERVER_PORT": "0",
            "CLEAR_LOGS": "1",
        },
    )
    try:
        _wait_started(proc)
        assert not sentinel.exists()
        assert log_dir.is_dir()
    finally:
        _stop(proc)
