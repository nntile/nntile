"""
Adapter that wraps nntile inference engines as gateway engines.

Importing this module does NOT import nntile/transformers — those imports
are deferred to NNTileModelLoader.load() so the rest of the gateway (config,
schemas, storage, auth, server with a fake registry) can be exercised
without StarPU or model weights.
"""

from typing import Protocol

from nntile_gateway.engine import (
    GatewayEngine,
    GenerateOptions,
    GenerateResult,
)
from nntile_gateway.schemas import ModelSpec


class ModelLoader(Protocol):
    def load(self, spec: ModelSpec) -> GatewayEngine: ...


class _NNTileEngineAdapter:
    """Wraps an LlmSyncInferenceEngine + tokenizer as a GatewayEngine."""

    def __init__(self, engine, tokenizer) -> None:
        self._engine = engine
        self._tokenizer = tokenizer

    def generate(self, prompt: str, options: GenerateOptions
                 ) -> GenerateResult:
        from nntile.model.generation.llm import (
            GenerationMode,
            GenerationParams,
        )

        params = GenerationParams(
            max_tokens=options.max_tokens,
            use_cache=True,
            need_static_padding=False,
            top_k=options.top_k,
            top_p_thr=options.top_p,
        )
        mode = GenerationMode.Greedy
        if options.top_k is not None:
            mode = GenerationMode.TopK
        elif options.top_p is not None:
            mode = GenerationMode.TopP

        prompt_ids = self._tokenizer(prompt)["input_ids"]
        prompt_tokens = len(prompt_ids)

        text = self._engine.generate(prompt, params=params, mode=mode)
        completion_text = text[len(prompt):] if text.startswith(prompt) else text
        completion_tokens = len(self._tokenizer(completion_text)["input_ids"])
        finish = (
            "length" if completion_tokens >= options.max_tokens else "stop"
        )
        return GenerateResult(
            text=completion_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish,
        )

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(
            tok, "chat_template", None
        ):
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback minimal template.
        rendered = []
        for m in messages:
            rendered.append(f"<|{m['role']}|>\n{m['content']}")
        rendered.append("<|assistant|>\n")
        return "\n".join(rendered)


class NNTileModelLoader:
    def load(self, spec: ModelSpec) -> GatewayEngine:
        from nntile.inference.llm_sync_engine import (
            LlmSyncInferenceEngine,
        )

        tokenizer = self._build_tokenizer(spec)
        model = self._build_model(spec)
        engine = LlmSyncInferenceEngine(model, tokenizer, spec.max_seq_len)
        return _NNTileEngineAdapter(engine, tokenizer)

    def _build_tokenizer(self, spec: ModelSpec):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            spec.hf_name, cache_dir=spec.cache_dir
        )

    def _build_model(self, spec: ModelSpec):
        if spec.family == "llama":
            from nntile.model.llama_causal import LlamaForCausalLM

            return LlamaForCausalLM.from_pretrained(
                spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
                **spec.extra,
            )
        if spec.family == "gpt2":
            from nntile.model.gpt2 import GPT2Model

            return GPT2Model.from_pretrained(
                spec.hf_name,
                spec.batch_size,
                spec.batch_size,
                spec.max_seq_len,
                cache_dir=spec.cache_dir,
            )
        if spec.family == "gpt_neo":
            from nntile.model.gpt_neo_causal import (
                GPTNeoForCausalLM,
            )

            return GPTNeoForCausalLM.from_pretrained(
                spec.batch_size,
                spec.batch_size,
                spec.max_seq_len,
                cache_dir=spec.cache_dir,
                remote_model_name=spec.hf_name,
                dtype=spec.dtype,
            )
        if spec.family == "gpt_neox":
            from nntile.model.gpt_neox_causal import (
                GPTNeoXForCausalLM,
            )

            return GPTNeoXForCausalLM.from_pretrained(
                spec.batch_size,
                spec.batch_size,
                spec.max_seq_len,
                cache_dir=spec.cache_dir,
                remote_model_name=spec.hf_name,
                dtype=spec.dtype,
            )
        if spec.family == "t5":
            from nntile.model.t5_model import T5ForConditionalGeneration

            # T5 is seq2seq: max_seq_len is the encoder static padding;
            # decoder seq len comes from extra.max_new_tokens (default 64).
            dec_seq_len = int(spec.extra.get("max_new_tokens", 64))
            return T5ForConditionalGeneration.from_pretrained(
                model_name=spec.hf_name,
                enc_seq_len=spec.max_seq_len,
                dec_seq_len=dec_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        raise ValueError(f"unsupported model family: {spec.family!r}")
