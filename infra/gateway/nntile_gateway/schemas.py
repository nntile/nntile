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
    id: str
    family: ModelFamily
    hf_name: str
    dtype: str
    max_seq_len: int
    status: Literal["loading", "ready", "error"]
    error: str | None = None


class CreateKeyRequest(BaseModel):
    name: str
    expires_at: float | None = Field(
        default=None, description="Unix timestamp; None = never")


class CreateKeyResponse(BaseModel):
    id: str
    name: str
    key: str = Field(description="Plaintext key, shown only once")


class KeyInfo(BaseModel):
    id: str
    name: str
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None


class OpenAIModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nntile"
    # Non-OpenAI extensions used by the bot to size requests sensibly
    # without round-tripping a 400 from the gateway.
    family: ModelFamily | None = None
    task: ModelTask | None = None
    max_seq_len: int | None = None


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModelObject]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    n: int = 1


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


# --- embeddings ------------------------------------------------------

class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int = 0


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingObject]
    usage: EmbeddingsUsage


# --- fill-mask --------------------------------------------------------

class FillMaskRequest(BaseModel):
    model: str
    input: str
    top_k: int = 5


class FillMaskCandidate(BaseModel):
    token: int
    token_str: str
    score: float
    sequence: str


class FillMaskUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class FillMaskResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    # Outer list is per [MASK] position in the input order.
    data: list[list[FillMaskCandidate]]
    usage: FillMaskUsage
