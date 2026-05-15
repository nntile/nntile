import uvicorn
from nntile_gateway.config import GatewayConfig
from nntile_gateway.server import build_app


def main() -> None:
    cfg = GatewayConfig()
    cfg.validate()
    # nntile.Context must be created in-process before any model load and
    # kept alive for the lifetime of the server. Without holding a name,
    # the temporary is GC'd immediately and StarPU is shut down before
    # the first request arrives.
    import nntile

    ctx = nntile.Context(  # noqa: F841 (held for lifetime)
        ncpu=cfg.ncpu, ncuda=cfg.ncuda, ooc=0, logger=0, verbose=0)
    try:
        app = build_app(cfg)
        uvicorn.run(app, host=cfg.host, port=cfg.port, workers=1)
    finally:
        nntile.starpu.wait_for_all()
        ctx.shutdown()


if __name__ == "__main__":
    main()
