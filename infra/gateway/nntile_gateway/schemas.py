"""Pydantic schemas for the gateway's HTTP wire format.

Three groups of types:

* **Admin** (`ModelSpec`, `ModelInfo`, `CreateKey*`, `KeyInfo`) drive
  the `/admin/*` routes used by an operator to register models and
  issue per-user API keys.
* **OpenAI-compatible** (`OpenAIModelObject`, `Completion*`, `Chat*`,
  `Embeddings*`) match the shapes a stock OpenAI client expects. We
  extend `OpenAIModelObject` with `family`/`task`/`max_seq_len` so
  the bot can size requests without round-tripping a 400.
* **Fill-mask** (`FillMask*`) is an nntile-specific endpoint shaped
  after `transformers.pipeline('fill-mask')`. Not in OpenAI."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelFamily = Literal[
    "llama", "gpt2", "gpt_neo", "gpt_neox", "t5", "bert", "roberta",
]

# Task = what the model is being registered to serve. For most families
# the task is fixed by the architecture (causal LMs do "completions"),
# but encoder-only families like bert/roberta can do either embeddings
# or fill-mask depending on whether we load BertModel or BertForMaskedLM.
ModelTask = Literal["completions", "embeddings", "fill_mask"]


class ModelSpec(BaseModel):
    """Admin payload for `POST /admin/models`.

    `id` is the public name clients use in `/v1/*` request bodies;
    `hf_name` is the HuggingFace repo (or local path) used to load the
    weights. `extra` carries family-specific knobs that the loader
    interprets (e.g. nntile tile sizes). `task=None` is resolved by
    `model_loader._resolve_task` from the family."""

    id: str = Field(description="Public model id, used in /v1/* requests")
    family: ModelFamily
    hf_name: str = Field(description="HuggingFace repo or local path")
    dtype: str = "fp32"
    max_seq_len: int = 1024
    batch_size: int = 1
    task: ModelTask | None = Field(
        default=None,
        description=(
            "Task to serve. If None, defaults to 'embeddings' for "
            "bert/roberta and 'completions' otherwise."),
    )
    cache_dir: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Admin view of a registered model.

    Returned by `POST /admin/models` (synchronous load result) and by
    `GET /admin/models`. `status="error"` carries the underlying
    exception text in `error` so the operator can drop or retry the
    spec via `DELETE /admin/models/{id}`."""

    id: str
    family: ModelFamily
    hf_name: str
    dtype: str
    max_seq_len: int
    status: Literal["loading", "ready", "error"]
    error: str | None = None


class CreateKeyRequest(BaseModel):
    """Admin payload for `POST /admin/keys` (issue a user API key)."""

    name: str
    expires_at: float | None = Field(
        default=None, description="Unix timestamp; None = never")


class CreateKeyResponse(BaseModel):
    """One-shot reply for `POST /admin/keys`.

    The plaintext `key` is shown exactly once; storage keeps only the
    SHA-256 hash, so the operator must capture this value at issuance
    or rotate the key."""

    id: str
    name: str
    key: str = Field(description="Plaintext key, shown only once")


class KeyInfo(BaseModel):
    """Admin listing entry. No plaintext key -- only metadata."""

    id: str
    name: str
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None


class OpenAIModelObject(BaseModel):
    """One entry in `GET /v1/models`.

    Shaped like the OpenAI Models API but with three extension fields
    (`family`, `task`, `max_seq_len`) so the bot can pick an
    appropriate `max_tokens` cap on `/select` without having to probe
    with an over-sized request. Stock OpenAI clients ignore unknown
    fields."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nntile"
    family: ModelFamily | None = None
    task: ModelTask | None = None
    max_seq_len: int | None = None


class OpenAIModelList(BaseModel):
    """Envelope for `GET /v1/models`."""

    object: Literal["list"] = "list"
    data: list[OpenAIModelObject]


class CompletionRequest(BaseModel):
    """`POST /v1/completions` payload (raw-prompt text completion).

    `max_tokens` is the total decoded-sequence cap; the gateway
    rejects values larger than the model's `max_seq_len` so a bad
    request can't drive nntile past its static allocation."""

    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1


class CompletionChoice(BaseModel):
    """One completion result inside `CompletionResponse.choices`."""

    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    """Token-count summary returned with completion responses."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """`POST /v1/completions` reply (OpenAI shape)."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    """One message in a chat-completion conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """`POST /v1/chat/completions` payload.

    Adapters apply each model's tokenizer `chat_template` if present;
    otherwise message contents are concatenated (no role tags), since
    non-chat models echo role-tag fallbacks as plain pattern text."""

    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1


class ChatCompletionChoice(BaseModel):
    """One chat result inside `ChatCompletionResponse.choices`."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    """`POST /v1/chat/completions` reply (OpenAI shape)."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


# --- embeddings ------------------------------------------------------

class EmbeddingsRequest(BaseModel):
    """`POST /v1/embeddings` payload.

    Accepts a single string or a list (matching the OpenAI shape);
    the server iterates the list and returns one `EmbeddingObject`
    per input."""

    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


class EmbeddingObject(BaseModel):
    """One mean-pooled embedding vector inside `EmbeddingsResponse`."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int = 0


class EmbeddingsUsage(BaseModel):
    """Token-count summary for embedding responses."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    """`POST /v1/embeddings` reply (OpenAI shape)."""

    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingObject]
    usage: EmbeddingsUsage


# --- fill-mask --------------------------------------------------------

class FillMaskRequest(BaseModel):
    """`POST /v1/fill_mask` payload.

    The input must contain at least one mask token. Both `[MASK]`
    (BERT-style) and `<mask>` (RoBERTa-style) are accepted -- the
    server rewrites whichever the model's tokenizer doesn't natively
    recognise."""

    model: str
    input: str
    top_k: int = 5


class FillMaskCandidate(BaseModel):
    """One predicted token for a single [MASK] position.

    Shape matches `transformers.pipeline('fill-mask')`: `token` is the
    vocab id, `token_str` is the decoded surface form, and `sequence`
    is the input string with this candidate substituted in."""

    token: int
    token_str: str
    score: float
    sequence: str


class FillMaskUsage(BaseModel):
    """Token-count summary for fill-mask responses."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class FillMaskResponse(BaseModel):
    """`POST /v1/fill_mask` reply.

    `data` is a list of inner lists, one per [MASK] position in the
    input order, each containing the top-k candidates sorted by
    descending score."""

    object: Literal["list"] = "list"
    model: str
    data: list[list[FillMaskCandidate]]
    usage: FillMaskUsage
