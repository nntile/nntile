"""Engine protocols and result types used by gateway endpoints.

The gateway dispatches by capability rather than family: a route checks
whether the loaded model's engine implements `generate` / `embed` /
`fill_mask` and 400s otherwise. The Protocol classes here document the
contracts each adapter in `model_loader` must satisfy."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class GenerateOptions:
    """Per-request sampling options passed to a `GatewayEngine`.

    `max_tokens` is the total generated-sequence cap; the server-side
    validator rejects values exceeding the model's static
    `max_seq_len` so an oversized request can't walk past nntile's
    pre-allocated attention/tile buffers (which would trip a StarPU
    assertion and abort the process)."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


@dataclass
class GenerateResult:
    """Outcome of one text-generation request.

    `text` is just the completion (the prompt prefix is stripped if the
    underlying engine echoes it). `finish_reason="length"` indicates
    the generator hit `options.max_tokens` rather than emitting EOS."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"] = "stop"


@dataclass
class EmbedResult:
    """One embedding vector + the token count consumed to produce it.

    The current encoder-family adapter mean-pools the encoder's final
    hidden state over non-pad positions; future engines may use the
    CLS token or some pooled head -- the contract is just "a finite
    float vector"."""

    embedding: list[float]
    prompt_tokens: int


@dataclass
class FillMaskCandidate:
    """One predicted token for a single [MASK] position.

    Mirrors the shape of huggingface.transformers.pipeline('fill-mask')
    entries -- `sequence` is the input string rendered with this
    candidate substituted in, so callers can show it verbatim."""

    token: int
    token_str: str
    score: float
    sequence: str


@dataclass
class FillMaskResult:
    """Top-k fill-mask predictions for every [MASK] position.

    `candidates` is a list of inner lists -- one per [MASK] occurrence
    in the input, in source order, so callers with multi-mask prompts
    can correlate predictions with positions."""

    candidates: list[list[FillMaskCandidate]]
    prompt_tokens: int


class GatewayEngine(Protocol):
    """Adapter contract for causal-LM / seq2seq inference engines.

    Anything that can answer `/v1/completions` and
    `/v1/chat/completions` must satisfy this protocol. The server
    routes a request here when the model's engine has a `generate`
    attribute -- typically the causal LM families (gpt2, gpt_neo,
    gpt_neox, llama) or the T5 seq2seq path."""

    def generate(self, prompt: str, options: GenerateOptions
                 ) -> GenerateResult:
        """Generate completion text for `prompt`. See `GenerateResult`."""

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Render OpenAI-style chat messages into a single prompt.

        Adapters with a real `chat_template` on their tokenizer use it
        as-is; for non-chat tokenizers (e.g. vanilla gpt2) the fallback
        is "raw completion" -- message contents joined by blank lines,
        no role tags -- since role-tag fallbacks just make the model
        echo the tag pattern."""


class EmbeddingEngine(Protocol):
    """Adapter contract for encoder-only inference engines.

    Anything answering `/v1/embeddings` must satisfy this protocol.
    Routed by the server when the model's engine has an `embed`
    attribute -- typically the encoder-only families (bert, roberta)
    registered with task='embeddings'."""

    def embed(self, text: str) -> EmbedResult:
        """Compute one embedding vector for `text`. See `EmbedResult`."""


class FillMaskEngine(Protocol):
    """Adapter contract for masked-LM fill-mask inference.

    Routed when the model's engine has a `fill_mask` attribute --
    typically bert/roberta registered with task='fill_mask'. The
    `top_k` argument bounds how many candidates per [MASK] position
    are returned; the adapter caps it at >= 1."""

    def fill_mask(self, text: str, top_k: int) -> FillMaskResult:
        """Return top-k predictions per [MASK]. See `FillMaskResult`."""
