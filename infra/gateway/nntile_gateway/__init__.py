"""Public surface of the nntile-gateway service package.

Importing `nntile_gateway` only pulls in config and the FastAPI app
factory; nothing here touches `nntile` itself, so this package can
be imported in a process that has no CUDA/StarPU access (e.g. tests
or the bot's own client)."""

from nntile_gateway.config import GatewayConfig
from nntile_gateway.server import build_app

__all__ = ["GatewayConfig", "build_app"]
