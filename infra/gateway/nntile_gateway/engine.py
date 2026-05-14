from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class GenerateOptions:
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"] = "stop"


class GatewayEngine(Protocol):
    def generate(self, prompt: str, options: GenerateOptions
                 ) -> GenerateResult: ...

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str: ...
