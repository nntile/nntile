import uvicorn

from nntile_gateway.config import GatewayConfig
from nntile_gateway.server import build_app


def main() -> None:
    cfg = GatewayConfig()
    cfg.validate()
    # nntile.Context must be created in-process before any model load.
    # We do it here so admin/model registration triggers loads under one
    # shared context, and the process must run with a single worker.
    import nntile  # noqa: F401  (kept for side-effect; user may rely on it)

    nntile.Context(  # type: ignore[attr-defined]
        ncpu=cfg.ncpu, ncuda=cfg.ncuda, ooc=0, logger=0, verbose=0)
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1)


if __name__ == "__main__":
    main()
