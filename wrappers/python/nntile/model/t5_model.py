# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_model.py
# T5 Model of NNTile Python package
#
# @version 1.1.0
# ruff: noqa: E501

import copy

import numpy as np
from transformers.models.t5.modeling_t5 import (
    T5Config as T5ConfigTorch,
    T5ForConditionalGeneration as T5ForConditionalGenerationTorch,
    T5ForSequenceClassification as T5ForSequenceClassificationTorch,
    T5Model as T5ModelTorch)

import nntile
import nntile.utils.constructors as nntc
from nntile.layer.embedding import Embedding
from nntile.layer.linear import Linear
from nntile.model.base_model import BaseModel
from nntile.model.t5_block import T5Stack
from nntile.model.t5_config import T5ConfigNNTile, T5EncoderDecoderConfig
from nntile.model.t5_lmhead import T5ClassificationHead
from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits)


class T5Model(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        encoder: T5Stack,
        decoder: T5Stack,
        encoder_config: T5ConfigNNTile,
        decoder_config: T5ConfigNNTile,
    ):
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.x = x
        self.decoder_x = decoder_x

        self.encoder = encoder
        self.decoder = decoder

        activations = (
            [x]
            + self.encoder.activations[1:]
            + [decoder_x]
            + self.decoder.activations[1:]
        )
        layers = self.encoder.layers + self.decoder.layers

        super().__init__(
            activations, layers, T5EncoderDecoderConfig(encoder_config, decoder_config)
        )

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ModelTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile,
    ):
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True

        encoder = T5Stack.from_torch(
            torch_model.encoder, x, encoder_config
        )
        encoder_output = encoder.activations[-1]
        decoder = T5Stack.from_torch(
            torch_model.decoder,
            decoder_x,
            decoder_config,
            encoder_output=encoder_output,
        )

        return cls(x, decoder_x, encoder, decoder, encoder_config, decoder_config)

    def to_torch(self):
        """Convert NNTile T5Model to PyTorch T5Model"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.encoder_config.d_model,
            d_ff=self.encoder_config.d_ff,
            num_layers=self.encoder_config.num_layers,
            num_decoder_layers=self.decoder_config.num_layers,
            num_heads=self.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
        )

        # Create PyTorch model
        torch_model = T5ModelTorch(torch_config)

        # Convert encoder and decoder
        torch_model.encoder = self.encoder.to_torch()
        torch_model.decoder = self.decoder.to_torch()

        return torch_model


class T5ForSequenceClassification(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        embedding_layer,
        embedding_layer_decoder,
        transformer: T5Model,
        lm_head: T5ClassificationHead,
    ):
        self.embedding = embedding_layer
        self.embedding_decoder = embedding_layer_decoder
        self.transformer = transformer
        self.classification_head = lm_head

        activations = (
            [x, decoder_x]
            + [self.embedding.activations_output[0]]
            + transformer.activations[1:]
            + lm_head.activations[1:]
        )
        layers = (
            [self.embedding, self.embedding_decoder]
            + transformer.layers
            + lm_head.layers
        )

        super().__init__(activations, layers, transformer.config)

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ForSequenceClassificationTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile
    ):
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "bf16": Tensor_bf16,
            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
            "fp32_fast_bf16": Tensor_fp32_fast_bf16,
        }

        tensor_type = dtype2tensor_type[config.dtype]

        embedding_layer = nntile.layer.embedding.Embedding.from_torch(
            torch_model.transformer.shared,
            x,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile
        )
        transformer = T5Model.from_torch(
            torch_model.transformer,
            embedding_layer.activations_output[0],
            embedding_layer.activations_output[0],
            config
        )
        lm_head = T5ClassificationHead.from_torch(
            torch_model.classification_head,
            transformer.activations[-1],
            config,
            torch_model.config.num_labels
        )

        return cls(
                x,
                decoder_x,
                embedding_layer,
                embedding_layer,
                transformer,
                lm_head
            )

    def to_torch(self):
        """Convert NNTile T5ForSequenceClassification
        to PyTorch T5ForSequenceClassification
        """
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.transformer.encoder_config.d_model,
            d_ff=self.transformer.encoder_config.d_ff,
            num_layers=self.transformer.encoder_config.num_layers,
            num_decoder_layers=self.transformer.decoder_config.num_layers,
            num_heads=self.transformer.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.transformer.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
            num_labels=self.classification_head.num_labels,
            decoder_start_token_id=0,
            pad_token_id=0,
        )

        # Create PyTorch model
        torch_model = T5ForSequenceClassificationTorch(torch_config)

        # Convert transformer and classification head
        torch_model.transformer = self.transformer.to_torch()
        torch_model.transformer.shared = self.embedding.to_torch()
        torch_model.transformer.encoder.embed_tokens = torch_model.transformer.shared
        torch_model.transformer.decoder.embed_tokens = torch_model.transformer.shared

        torch_model.classification_head = self.classification_head.to_torch()

        return torch_model


class T5ForConditionalGeneration(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        embedding_layer,
        embedding_layer_decoder,
        transformer: T5Model,
        lm_head: Linear
    ):
        self.embedding = embedding_layer
        self.embedding_decoder = embedding_layer_decoder
        self.transformer = transformer
        self.lm_head = lm_head

        # Bind named handles to the int64 token-id inputs so generate()
        # can rewrite them in place between forward passes. For the
        # training example x and decoder_x are the same TensorMoments;
        # for inference (see from_pretrained) they are distinct tensors.
        self.x_enc = x
        self.x_dec = decoder_x
        enc_config = transformer.encoder_config
        self.eos_token_id = enc_config.eos_token_id
        self.decoder_start_token_id = enc_config.decoder_start_token_id
        self.pad_token_id = enc_config.pad_token_id

        activations = (
            [x, decoder_x]
            + [self.embedding.activations_output[0]]
            + transformer.activations[1:]
            + [lm_head.activations_output[0]]
        )
        layers = (
            [self.embedding, self.embedding_decoder]
            + transformer.layers
            + [lm_head]
        )

        super().__init__(activations, layers, transformer.config)

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ForConditionalGenerationTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile
    ):
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "bf16": Tensor_bf16,
            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
            "fp32_fast_bf16": Tensor_fp32_fast_bf16,
        }

        tensor_type = dtype2tensor_type[config.dtype]

        embedding_layer = nntile.layer.embedding.Embedding.from_torch(
            torch_model.shared,
            x,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile,
        )
        transformer = T5Model.from_torch(
            torch_model,
            embedding_layer.activations_output[0],
            embedding_layer.activations_output[0],
            config
        )
        print(f"transformer.activations[-1].value.basetile_shape: {transformer.activations[-1].value.basetile_shape}")
        lm_head = Linear.from_torch(
            torch_model.lm_head,
            transformer.activations[-1],
            torch_model.lm_head.out_features,
            config.redux
        )

        return cls(
                x,
                decoder_x,
                embedding_layer,
                embedding_layer,
                transformer,
                lm_head
            )

    @classmethod
    def from_torch_for_inference(
        cls,
        torch_model: T5ForConditionalGenerationTorch,
        config: T5ConfigNNTile,
        seq_len: int,
        batch_size: int = 1,
        seq_len_tile: int = None,
        batch_size_tile: int = None,
    ):
        """Build a T5ForConditionalGeneration whose encoder and decoder
        consume independent int64 token tensors.

        Unlike from_torch (which wires the same embedded tensor into both
        the encoder and decoder for the teacher-forcing training example),
        this wiring is what's needed to actually run seq2seq inference:
        a fixed source sequence on the encoder side, and a growing target
        prefix on the decoder side, both updatable independently.

        Note: encoder and decoder are sized to the same seq_len because
        T5Attention.generate_simple (see layer/t5_attention.py) takes the
        K/V output shape from x_q rather than x_k, so cross-attention
        breaks if enc_seq_len != dec_seq_len. Pick seq_len large enough
        to hold both the source and the longest generation."""
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "bf16": Tensor_bf16,
            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
            "fp32_fast_bf16": Tensor_fp32_fast_bf16,
        }
        tensor_type = dtype2tensor_type[config.dtype]
        seq_len_tile = seq_len_tile or seq_len
        batch_size_tile = batch_size_tile or batch_size

        # Encoder input: int64 tokens, pre-filled with pad so an unset
        # generate() call sees a well-defined state.
        enc_traits = TensorTraits(
            [seq_len, batch_size],
            [seq_len_tile, batch_size_tile],
        )
        enc_distr = [0] * enc_traits.grid.nelems
        x_enc = Tensor_int64(enc_traits, enc_distr)
        x_enc.from_array(np.full(
            (seq_len, batch_size),
            config.pad_token_id,
            dtype=np.int64, order='F',
        ))
        x_enc_tm = TensorMoments(x_enc, None, False)

        # Decoder input: starts with [decoder_start_token_id, pad, pad, ...].
        dec_traits = TensorTraits(
            [seq_len, batch_size],
            [seq_len_tile, batch_size_tile],
        )
        dec_distr = [0] * dec_traits.grid.nelems
        x_dec = Tensor_int64(dec_traits, dec_distr)
        dec_init = np.full(
            (seq_len, batch_size),
            config.pad_token_id,
            dtype=np.int64, order='F',
        )
        dec_init[0, :] = config.decoder_start_token_id
        x_dec.from_array(dec_init)
        x_dec_tm = TensorMoments(x_dec, None, False)

        # Two embedding layers from the same shared torch weights. They
        # share *values* but each owns its own NNTile w tensor; that's
        # fine for inference (no gradients to accumulate).
        enc_embedding = Embedding.from_torch(
            torch_model.shared, x_enc_tm,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile,
        )
        dec_embedding = Embedding.from_torch(
            torch_model.shared, x_dec_tm,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile,
        )

        transformer = T5Model.from_torch(
            torch_model,
            enc_embedding.activations_output[0],
            dec_embedding.activations_output[0],
            config,
        )
        lm_head = Linear.from_torch(
            torch_model.lm_head,
            transformer.activations[-1],
            torch_model.lm_head.out_features,
            config.redux,
        )
        return cls(
            x_enc_tm, x_dec_tm,
            enc_embedding, dec_embedding,
            transformer, lm_head,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        seq_len: int,
        batch_size: int = 1,
        seq_len_tile: int = None,
        batch_size_tile: int = None,
        d_model_tile: int = None,
        d_ff_tile: int = None,
        n_head_tile: int = None,
        dtype: str = "fp32",
        cache_dir: str = None,
    ):
        """Load a HuggingFace T5ForConditionalGeneration checkpoint and
        convert it to NNTile with the inference-time wiring.

        Constraints to be aware of (these are nntile-side limitations,
        not fundamental to T5):
          * Only the gated T5 FF variant is supported (T5 v1.1 /
            Flan-T5). Vanilla t5-small / t5-base use ungated FF and
            raise ValueError here.
          * Encoder and decoder are both sized to `seq_len`. The cross
            attention K/V are sized from the Q sequence (see
            layer/t5_attention.generate_simple), so enc_seq_len must
            equal dec_seq_len today.
          * The encoder has no padding attention mask. If `seq_len` is
            much larger than the actual source token count, encoder
            self-attention attends to padding tokens uniformly and the
            hidden states get diluted, degrading output quality. Pick
            `seq_len` close to the actual source length, or pad the
            input with EOS rather than PAD, for usable output.
          * Generation is greedy argmax only (mode argument is ignored
            in generate()); no top-k/top-p sampling and no KV cache."""
        torch_model = T5ForConditionalGenerationTorch.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=False,
        )
        torch_model.eval()
        torch_config = torch_model.config

        # nntile's T5 FF implementation (T5DenseGatedActDense, see
        # nntile/model/t5_ff.py) only handles the gated variant used by
        # T5 v1.1 / Flan-T5 (feed_forward_proj == 'gated-gelu' or
        # 'gated-relu'). Vanilla t5-small / t5-base use ungated FF
        # ('relu'), which has 2 weight matrices instead of 3, and the
        # index-based weight copy would silently misalign.
        ffp = getattr(torch_config, "feed_forward_proj", "relu")
        if not ffp.startswith("gated-"):
            raise ValueError(
                f"{model_name!r} uses feed_forward_proj={ffp!r}, but "
                "nntile only implements the gated T5 FF variant. Try a "
                "T5 v1.1 / Flan-T5 checkpoint instead "
                "(e.g. google/flan-t5-small, google/t5-v1_1-small).")

        config = T5ConfigNNTile(
            vocab_size=torch_config.vocab_size,
            d_model=torch_config.d_model,
            d_model_tile=d_model_tile or torch_config.d_model,
            d_ff=torch_config.d_ff,
            d_ff_tile=d_ff_tile or torch_config.d_ff,
            d_kv=torch_config.d_kv,
            d_kv_tile=torch_config.d_kv,
            num_layers=torch_config.num_layers,
            n_head=torch_config.num_heads,
            n_head_tile=n_head_tile or torch_config.num_heads,
            dropout_rate=0.0,
            layer_norm_epsilon=torch_config.layer_norm_epsilon,
            dtype=dtype,
            eos_token_id=torch_config.eos_token_id,
            decoder_start_token_id=torch_config.decoder_start_token_id,
            pad_token_id=torch_config.pad_token_id,
        )
        return cls.from_torch_for_inference(
            torch_model, config,
            seq_len=seq_len,
            batch_size=batch_size,
            seq_len_tile=seq_len_tile,
            batch_size_tile=batch_size_tile,
        )

    def generate(
        self,
        input_ids,
        prefill_size: int,
        params,
        mode=None,
    ):
        """Static seq2seq generation. Matches the protocol expected by
        nntile.inference.llm_sync_engine.LlmSyncInferenceEngine.

        Re-runs the full encoder+decoder on every step (no KV cache).
        That's O(max_tokens * (enc + dec) flops), so suitable for short
        replies; KV caching would require finishing the dynamic-decode
        path in t5_attention which is a much larger change.

        Returns (output_ids, effective_size) where output_ids is a
        Tensor_int64 of shape (effective_size, 1) containing only the
        generated tokens (the decoder_start_token is not included)."""
        enc_seq_len, batch_size = self.x_enc.value.shape
        dec_seq_len = self.x_dec.value.shape[0]

        # Copy the source tokens into the encoder slot. The caller may
        # pass an already-padded tensor (need_static_padding=True) or a
        # variable-length one; we always pad/truncate to enc_seq_len.
        src_np = nntc.to_numpy(input_ids).astype(np.int64)
        if src_np.ndim == 1:
            src_np = src_np.reshape(-1, 1)
        actual_src_len = min(src_np.shape[0], enc_seq_len)
        src_padded = np.full(
            (enc_seq_len, batch_size),
            self.pad_token_id,
            dtype=np.int64, order='F',
        )
        src_padded[:actual_src_len, :] = src_np[:actual_src_len, :batch_size]
        self.x_enc.value.from_array(src_padded)

        # Decoder slot: start with [decoder_start_token_id, pad, pad, ...]
        # and grow it one token per step.
        dec_np = np.full(
            (dec_seq_len, batch_size),
            self.pad_token_id,
            dtype=np.int64, order='F',
        )
        dec_np[0, :] = self.decoder_start_token_id

        max_tokens = min(params.max_tokens, dec_seq_len - 1)
        generated = []
        for p in range(max_tokens):
            self.x_dec.value.from_array(dec_np)
            self.forward_async()
            # nntc.to_numpy blocks until the lm_head output is on host.
            logits_np = nntc.to_numpy(self.activations[-1].value)
            # Logits shape: (vocab_size, dec_seq_len, batch_size).
            # Position p's logits predict the token at position p+1.
            pred = int(np.argmax(logits_np[:, p, 0]))
            if pred == self.eos_token_id:
                break
            dec_np[p + 1, 0] = pred
            generated.append(pred)

        if not generated:
            out_np = np.array(
                [[self.decoder_start_token_id]],
                dtype=np.int64, order='F',
            )
            out_tensor = nntc.from_array(out_np)
            return out_tensor, 1
        out_np = np.asarray(generated, dtype=np.int64).reshape(-1, 1)
        out_np = np.asfortranarray(out_np)
        out_tensor = nntc.from_array(out_np)
        return out_tensor, len(generated)

    def to_torch(self):
        """Convert NNTile T5ForConditionalGeneration
        to PyTorch T5ForConditionalGeneration
        """
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.transformer.encoder_config.d_model,
            d_ff=self.transformer.encoder_config.d_ff,
            num_layers=self.transformer.encoder_config.num_layers,
            num_decoder_layers=self.transformer.decoder_config.num_layers,
            num_heads=self.transformer.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.transformer.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
            decoder_start_token_id=0,
            pad_token_id=0,
        )

        # Create PyTorch model
        torch_model = T5ForConditionalGenerationTorch(torch_config)

        # Convert transformer and classification head
        transformer = self.transformer.to_torch()
        torch_model.encoder = transformer.encoder
        torch_model.decoder = transformer.decoder
        torch_model.shared = self.embedding.to_torch()
        torch_model.encoder.embed_tokens = torch_model.shared
        torch_model.decoder.embed_tokens = torch_model.shared
        torch_model.lm_head = self.lm_head.to_torch()

        return torch_model
