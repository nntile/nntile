import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GATEWAY_ROOT = Path(__file__).resolve().parents[1]
if str(GATEWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(GATEWAY_ROOT))

# E402: the imports below intentionally come after the sys.path insert
# so that running pytest from the repo root finds the in-tree package
# instead of any installed copy.
from nntile_gateway.config import GatewayConfig  # noqa: E402
from nntile_gateway.engine import GenerateOptions, GenerateResult  # noqa: E402
from nntile_gateway.model_loader import ModelLoader  # noqa: E402
from nntile_gateway.schemas import ModelSpec  # noqa: E402
from nntile_gateway.server import build_app  # noqa: E402
from nntile_gateway.storage.memory import InMemoryStorage  # noqa: E402

ADMIN_TOKEN = "test-admin-token-12345"


class FakeEngine:
    """Deterministic engine for tests; no nntile dependency."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.calls: list[tuple[str, GenerateOptions]] = []

    def generate(self, prompt: str, options: GenerateOptions
                 ) -> GenerateResult:
        self.calls.append((prompt, options))
        completion = f" [echo:{self.spec.id}]"
        return GenerateResult(
            text=completion,
            prompt_tokens=len(prompt.split()),
            completion_tokens=2,
            finish_reason="stop",
        )

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(parts) + "\nassistant:"


class FakeLoader(ModelLoader):
    def __init__(self) -> None:
        self.engines: dict[str, FakeEngine] = {}
        self.fail_on: set[str] = set()

    def load(self, spec: ModelSpec):
        if spec.id in self.fail_on:
            raise RuntimeError(f"forced load failure for {spec.id}")
        engine = FakeEngine(spec)
        self.engines[spec.id] = engine
        return engine


@pytest.fixture
def config() -> GatewayConfig:
    os.environ.pop("NNTILE_ADMIN_TOKEN", None)
    return GatewayConfig(
        admin_token=ADMIN_TOKEN,
        host="127.0.0.1",
        port=0,
        storage="memory",
        sqlite_path="",
        auth_cache_ttl=60,
        auth_cache_size=64,
        ncpu=-1,
        ncuda=-1,
    )


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def loader() -> FakeLoader:
    return FakeLoader()


@pytest.fixture
def client(config, storage, loader):
    app = build_app(config, storage=storage, loader=loader)
    with TestClient(app) as c:
        yield c


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live integration tests that load real nntile models",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(
        reason="live tests are opt-in; pass --live to enable")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
