# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/_platform.py
# Optional nntile-server client when NNTILE_SERVER_SOCKET is set.

"""Flush recorded TensorGraphs to the closed nntile-server.

Loaded only when ``NNTILE_SERVER_SOCKET`` is set. Speaks
``nntile_protocol`` (platform package). Does not start StarPU.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import Any

from . import _C

_tls = threading.local()


class ClientError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class _Conn:
    def __init__(self, path: str, account_id: str) -> None:
        from nntile_protocol import PROTOCOL_VERSION, nntile_revision, recv, send

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        send(
            self.sock,
            {
                "type": "Handshake",
                "protocol_version": PROTOCOL_VERSION,
                "nntile_revision": nntile_revision(),
                "role": "kernel",
                "account_id": account_id,
            },
        )
        reply = recv(self.sock)
        if reply.get("type") != "HandshakeOk":
            raise ClientError(
                reply.get("code", "Protocol"),
                reply.get("message", str(reply)),
            )
        self.session_id = reply["session_id"]
        self.lock = threading.Lock()

    def rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        from nntile_protocol import recv, send

        with self.lock:
            send(self.sock, msg)
            reply = recv(self.sock)
        if reply.get("type") == "Error":
            raise ClientError(
                reply.get("code", "Internal"),
                reply.get("message", ""),
            )
        if reply.get("type") == "HandshakeReject":
            raise ClientError(
                reply.get("code", "Protocol"),
                reply.get("message", ""),
            )
        return reply


def _state() -> _Conn:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        raise RuntimeError("torch_nntile.init_context() was not called")
    return conn


def _ingress(
    node_id: int,
    payload: bytes,
    shape: list[int],
    dtype: str,
) -> None:
    from nntile_protocol.shm import write_bytes

    conn = _state()
    blob = bytes(payload)
    shm = write_bytes(blob, prefix="nntile_ing")
    conn.rpc(
        {
            "type": "Ingress",
            "session_id": conn.session_id,
            "node_id": int(node_id),
            "shape": [int(x) for x in shape],
            "dtype": str(dtype),
            "shm_name": shm,
            "nbytes": len(blob),
        }
    )


def _flush(
    phase_json: str,
    gather_ids: list[int],
    wait_only: bool,
) -> bytes:
    from nntile_protocol.shm import read_bytes

    conn = _state()
    phase = json.loads(phase_json) if phase_json else {"nodes": [], "ops": []}
    reply = conn.rpc(
        {
            "type": "Flush",
            "session_id": conn.session_id,
            "phase_ir": phase,
            "gather_node_ids": [int(n) for n in gather_ids],
            "wait_only": bool(wait_only),
            "invalidate_node_ids": [],
        }
    )
    if wait_only:
        return b""
    gathered = reply.get("gathered") or []
    if not gathered:
        return b""
    g = gathered[0]
    return read_bytes(g["shm_name"], int(g["nbytes"]))


def connect(account_id: str | None = None) -> None:
    """Handshake with nntile-server and install host-copy hooks."""
    path = os.environ.get("NNTILE_SERVER_SOCKET", "").strip()
    if not path:
        raise RuntimeError("NNTILE_SERVER_SOCKET is not set")
    try:
        import nntile_protocol  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "NNTILE_SERVER_SOCKET is set but nntile_protocol is not "
            "installed; the platform kernel needs that package"
        ) from exc
    if getattr(_tls, "conn", None) is not None:
        _C.set_platform_hooks(_ingress, _flush)
        return
    account = account_id or os.environ.get("NNTILE_ACCOUNT_ID", "alice")
    _tls.conn = _Conn(path, account)
    _C.set_platform_hooks(_ingress, _flush)


def teardown() -> None:
    conn = getattr(_tls, "conn", None)
    if conn is None:
        _C.clear_platform_hooks()
        return
    try:
        conn.rpc(
            {
                "type": "Teardown",
                "session_id": conn.session_id,
            }
        )
    except Exception:
        pass
    try:
        conn.sock.close()
    except Exception:
        pass
    _tls.conn = None
    _C.clear_platform_hooks()
