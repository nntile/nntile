"""
Adapter that wraps nntile inference engines as gateway engines.

Importing this module does NOT import nntile/transformers — those imports
are deferred to NNTileModelLoader.load() so the rest of the gateway (config,
schemas, storage, auth, server with a fake registry) can be exercised
without StarPU or model weights.
"""

from typing import Protocol, Union

from nntile_gateway.engine import (
    EmbeddingEngine,
    EmbedResult,
    FillMaskCandidate,
    FillMaskEngine,
    FillMaskResult,
    GatewayEngine,
    GenerateOptions,
    GenerateResult,
)
from nntile_gateway.schemas import ModelSpec


class ModelLoader(Protocol):
    def load(
        self, spec: ModelSpec
    ) -> Union[GatewayEngine, EmbeddingEngine, FillMaskEngine]: ...


def _build_padding_mask(seq_len: int, actual_len: int):
    """Boolean (seq_len, seq_len) F-order array suitable for the BERT
    self-attention mask. mask[k, q] = True iff the key position k is a
    real (non-pad) token; all queries see the same column mask."""
    import numpy as np

    keep = np.zeros(seq_len, dtype=bool)
    keep[:actual_len] = True
    mask = np.broadcast_to(keep[:, None], (seq_len, seq_len)).copy()
    return np.asfortranarray(mask)


def _apply_padding_mask(model, seq_len: int, actual_len: int) -> None:
    """Walk the model's layers, find every BertSelfAttention, and rewrite
    its mask tensor to mask out pad positions. Without this the encoder
    attends to padding uniformly and output quality drops sharply for
    short inputs (see test_live_bert results)."""
    from nntile.layer.bert_selfattention import BertSelfAttention

    mask_np = _build_padding_mask(seq_len, actual_len)
    for layer in model.layers:
        if isinstance(layer, BertSelfAttention):
            layer.mask.from_array(mask_np)


def _find_position_ids_tensor(model):
    """Locate the 1D (length=seq_len) int64 tensor used as input to the
    BertEmbeddings position-embedding layer, traversing the known module
    paths: BertModel.bert_embed, BertForMaskedLM.bert.bert_embed,
    RobertaModel.bert_embed, RobertaForMaskedLM.roberta.bert_embed."""
    paths = (
        ("bert_embed",),
        ("bert", "bert_embed"),
        ("roberta", "bert_embed"),
    )
    for path in paths:
        obj = model
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            continue
        pos_embed = getattr(obj, "pos_embed", None)
        if pos_embed is None:
            continue
        return getattr(pos_embed, "x", None)
    return None


def _apply_roberta_position_ids(
    model, padded_input_ids, pad_token_id: int,
) -> None:
    """Rewrite the position-id tensor using RoBERTa's offset convention.

    HF's RoBERTa does:
        mask = (input_ids != pad_idx)
        position_ids = (cumsum(mask) * mask) + pad_idx
    so for input_ids = [0, 31414, ..., 1, 1, 1] with pad_idx=1, you get
    positions = [2, 3, ..., 1, 1, 1] -- non-pad starts at pad_idx+1.

    nntile's BertEmbeddings.from_torch initialises pos_ids to
    arange(seq_len), which is BERT-correct but RoBERTa-incorrect. Without
    this override the model still picks the same top token, but
    probability mass is materially different from HF's reference."""
    import numpy as np

    pos_tensor = _find_position_ids_tensor(model)
    if pos_tensor is None:
        return  # nothing we can update; caller will get arange behaviour
    seq_len = padded_input_ids.shape[0]
    flat_ids = padded_input_ids[:, 0]
    mask = (flat_ids != pad_token_id).astype(np.int64)
    positions = (np.cumsum(mask) * mask) + pad_token_id
    pos_tensor.from_array(
        np.asarray(positions, dtype=np.int64, order='F'))


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


class _NNTileEmbeddingAdapter:
    """Wraps an encoder-only NNTile model (BertModel/RobertaModel) +
    tokenizer to satisfy the EmbeddingEngine protocol.

    Returns a single mean-pooled vector per input, computed over the
    non-pad token positions of the encoder's final hidden state. That
    pooling is what mitigates the missing-encoder-mask issue: the model
    still attends to pad positions internally (degrading values at all
    positions), but pooling over non-pad positions roughly matches the
    HuggingFace reference with the same (no-mask) input."""

    def __init__(
        self, model, tokenizer, seq_len: int, pad_token_id: int,
        family: str,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._pad_token_id = pad_token_id
        self._family = family

    def embed(self, text: str) -> EmbedResult:
        import numpy as np

        import nntile.utils.constructors as nntc

        ids = self._tokenizer(text)["input_ids"]
        if len(ids) > self._seq_len:
            ids = ids[: self._seq_len]
        actual_len = len(ids)

        padded = np.full(
            (self._seq_len, 1), self._pad_token_id,
            dtype=np.int64, order='F',
        )
        padded[:actual_len, 0] = ids
        _apply_padding_mask(self._model, self._seq_len, actual_len)
        if self._family == "roberta":
            _apply_roberta_position_ids(
                self._model, padded, self._pad_token_id)
        self._model.activations[0].value.from_array(padded)
        self._model.forward_async()
        hidden = nntc.to_numpy(self._model.activations[-1].value)
        # hidden: (hidden_size, seq_len, batch). Mean over real tokens.
        pooled = hidden[:, :actual_len, 0].mean(axis=1)
        return EmbedResult(
            embedding=pooled.astype(np.float32).tolist(),
            prompt_tokens=actual_len,
        )


class _NNTileFillMaskAdapter:
    """Wraps BertForMaskedLM / RobertaForMaskedLM for fill-mask.

    Returns top-k candidates per [MASK] position. Matches the shape
    of huggingface.transformers.pipeline('fill-mask'): each candidate
    has token id, decoded token string, softmax probability, and a
    rendered 'sequence' with the candidate substituted in."""

    def __init__(
        self, model, tokenizer, seq_len: int, pad_token_id: int,
        family: str,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._pad_token_id = pad_token_id
        self._family = family
        if getattr(tokenizer, "mask_token_id", None) is None:
            raise ValueError(
                f"tokenizer {tokenizer.__class__.__name__!r} has no "
                "mask_token_id; fill_mask is unsupported")
        self._mask_token_id = tokenizer.mask_token_id

    def fill_mask(self, text: str, top_k: int) -> FillMaskResult:
        import numpy as np

        import nntile.utils.constructors as nntc

        ids = list(self._tokenizer(text)["input_ids"])
        if len(ids) > self._seq_len:
            ids = ids[: self._seq_len]
        actual_len = len(ids)
        mask_positions = [
            i for i, t in enumerate(ids) if t == self._mask_token_id]
        if not mask_positions:
            raise ValueError(
                "input contains no [MASK] token; nothing to fill")

        padded = np.full(
            (self._seq_len, 1), self._pad_token_id,
            dtype=np.int64, order='F',
        )
        padded[:actual_len, 0] = ids
        _apply_padding_mask(self._model, self._seq_len, actual_len)
        if self._family == "roberta":
            _apply_roberta_position_ids(
                self._model, padded, self._pad_token_id)
        self._model.activations[0].value.from_array(padded)
        self._model.forward_async()
        logits = nntc.to_numpy(self._model.activations[-1].value)
        # logits: (vocab_size, seq_len, batch).

        out_per_mask: list[list[FillMaskCandidate]] = []
        top_k = max(1, top_k)
        for pos in mask_positions:
            v = logits[:, pos, 0]
            v = v - v.max()
            probs = np.exp(v)
            probs /= probs.sum()
            top_idx = np.argsort(probs)[::-1][:top_k]
            cands: list[FillMaskCandidate] = []
            for t_id in top_idx:
                t_int = int(t_id)
                token_str = self._tokenizer.decode([t_int]).strip()
                # Build the filled sequence by substituting at this pos.
                filled = list(ids)
                filled[pos] = t_int
                sequence = self._tokenizer.decode(
                    filled, skip_special_tokens=False)
                cands.append(FillMaskCandidate(
                    token=t_int,
                    token_str=token_str,
                    score=float(probs[t_int]),
                    sequence=sequence,
                ))
            out_per_mask.append(cands)

        return FillMaskResult(
            candidates=out_per_mask, prompt_tokens=actual_len)


_ENCODER_ONLY_FAMILIES = {"bert", "roberta"}


def _resolve_task(spec: ModelSpec) -> str:
    if spec.task is not None:
        return spec.task
    if spec.family in _ENCODER_ONLY_FAMILIES:
        return "embeddings"
    return "completions"


class NNTileModelLoader:
    def load(self, spec: ModelSpec):
        task = _resolve_task(spec)
        tokenizer = self._build_tokenizer(spec)

        if spec.family in _ENCODER_ONLY_FAMILIES:
            if task == "embeddings":
                model = self._build_encoder_model(spec)
                pad = self._pad_token_id(tokenizer)
                return _NNTileEmbeddingAdapter(
                    model, tokenizer, spec.max_seq_len, pad, spec.family)
            if task == "fill_mask":
                model = self._build_masked_lm_model(spec)
                pad = self._pad_token_id(tokenizer)
                return _NNTileFillMaskAdapter(
                    model, tokenizer, spec.max_seq_len, pad, spec.family)
            raise ValueError(
                f"family={spec.family!r} does not support task={task!r}")

        if task != "completions":
            raise ValueError(
                f"family={spec.family!r} only supports task='completions', "
                f"got task={task!r}")

        from nntile.inference.llm_sync_engine import (
            LlmSyncInferenceEngine,
        )

        model = self._build_model(spec)
        engine = LlmSyncInferenceEngine(model, tokenizer, spec.max_seq_len)
        return _NNTileEngineAdapter(engine, tokenizer)

    @staticmethod
    def _pad_token_id(tokenizer) -> int:
        return tokenizer.pad_token_id or 0

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

            # T5 is seq2seq. Encoder and decoder are sized to the same
            # seq_len because nntile's T5 cross-attention currently
            # requires enc_seq_len == dec_seq_len (see
            # nntile/layer/t5_attention.py: K/V shapes come from x_q).
            return T5ForConditionalGeneration.from_pretrained(
                model_name=spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        if spec.family in _ENCODER_ONLY_FAMILIES:
            # Handled by _build_encoder_model / _build_masked_lm_model
            # because the class depends on task, not just family.
            raise ValueError(
                f"call _build_encoder_model/_build_masked_lm_model for "
                f"family={spec.family!r}")
        raise ValueError(f"unsupported model family: {spec.family!r}")

    def _build_encoder_model(self, spec: ModelSpec):
        if spec.family == "bert":
            from nntile.model.bert import BertModel

            return BertModel.from_pretrained(
                model_name=spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        if spec.family == "roberta":
            from nntile.model.roberta import RobertaModel

            return RobertaModel.from_pretrained(
                model_name=spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        raise ValueError(
            f"_build_encoder_model: unsupported family {spec.family!r}")

    def _build_masked_lm_model(self, spec: ModelSpec):
        if spec.family == "bert":
            from nntile.model.bert import BertForMaskedLM

            return BertForMaskedLM.from_pretrained(
                model_name=spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        if spec.family == "roberta":
            from nntile.model.roberta import RobertaForMaskedLM

            return RobertaForMaskedLM.from_pretrained(
                model_name=spec.hf_name,
                seq_len=spec.max_seq_len,
                batch_size=spec.batch_size,
                dtype=spec.dtype,
                cache_dir=spec.cache_dir,
            )
        raise ValueError(
            f"_build_masked_lm_model: unsupported family {spec.family!r}")
