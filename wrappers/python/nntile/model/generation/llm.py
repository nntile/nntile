# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/generation/llm.py
#
# @version 1.1.0

import numpy as np

import nntile
import nntile.utils.constructors as nntc
from nntile.layer.cache_utils import KVCacheStorage
from nntile.model.generation.llm_beamsearch import generate_parallel
from nntile.model.generation.llm_params import GenerationMode, GenerationParams
from nntile.model.generation.llm_samplers import get_sampler
from nntile.tensor import Tensor
from nntile.utils import constructors as nnt_constructors
import torch.nn.functional as F

class EncoderDecoderGenerationMixin:
    def generate(
        self,
        encoder_input_ids: Tensor,
        decoder_input_ids: Tensor = None,
        params: GenerationParams = None,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        if params is None:
            params = GenerationParams()
        
        sampler = get_sampler(mode, params)
        
        # Run encoder once to get encoder hidden states
    #             nntile.functions.copy_async(x, self.activations[0].value)

    # def get_output(self) -> Tensor:
    #     return self.activations[-1].value
        
        print("SHAPES: ", encoder_input_ids.shape, self.embedding.activations_input[0].shape)
        # nntile.functions.copy_async(encoder_input_ids, self.embedding.activations_input[0])
        
        # if encoder_input_ids.shape[0] < self.embedding.activations_input[0].shape[0]:
            
        #     encoder_input_ids = F.pad(encoder_input_ids, (self.embedding.activations_input[0].shape[0] - encoder_input_ids.shape[0], 0))
        
        nntile.functions.copy_async(encoder_input_ids, self.embedding.activations_input[0])
        self.embedding.forward_async()
        self.transformer.encoder.forward_async()
        # encoder_hidden_states = self.activations[0].value
        
        # Initialize decoder input if not provided
        if decoder_input_ids is None:
            # Create a tensor with just the start token
            decoder_input_ids = nntc.from_array(
                np.array(
                    [
                        [self.config.decoder_config.decoder_start_token_id]
                        +[0]*(self.embedding.activations_input[0].shape[0]-1)
                    ]
                    , dtype=np.int64, order='F'
                ).T
            )
            
        output_ids = generate_autoregress_encoder_decoder(
            model=self,
            input_ids=decoder_input_ids,
            prefill_size=1,
            max_tokens=params.max_tokens,
            eos_token_id=self.config.decoder_config.eos_token_id,
        )
        return output_ids

class LLMGenerationMixin:
    def generate(
        self,
        input_ids: Tensor,
        prefill_size: int,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        sampler = get_sampler(mode, params)
        if params.need_static_padding:
            # This path only for compatibility with statically defined
            # model and not efficient on small examples
            if params.use_cache:
                raise Exception("No support for kvcache for static inference")
            if params.num_beams > 1:
                raise Exception(
                    "No support for beam search in static inference"
                )
            output_ids = generate_autoregress(
                model=self,
                input_ids=input_ids,
                prefill_size=prefill_size,
                max_tokens=params.max_tokens,
                eos_token_id=self.eos_token_id,
            )
        else:
            if params.num_beams == 1:
                output_ids = generate_autoregress_dynamic(
                    model=self,
                    input_ids=input_ids,
                    max_tokens=params.max_tokens,
                    eos_token_id=self.eos_token_id,
                    use_cache=params.use_cache,
                    sampler=sampler,
                )
            else:
                output_ids = generate_parallel(
                    model=self,
                    input_ids=input_ids,
                    max_tokens=params.max_tokens,
                    eos_token_id=self.eos_token_id,
                    num_beams=params.num_beams,
                    sampler=sampler,
                    sampling_mode=params.parallel_sampling_mode,
                )

        return output_ids

    async def generate_async(
        self,
        input_ids: Tensor,
        prefill_size: int,
        params: GenerationParams,
        mode: GenerationMode = GenerationMode.Greedy,
    ):
        sampler = get_sampler(mode, params)

        assert (
            params.num_beams == 1
        ), "No support for beam search in async inference"

        if params.need_static_padding:
            raise Exception("No support for async static inference")
        else:
            output_ids = await generate_autoregress_dynamic_async(
                self, input_ids, params.max_tokens,
                self.eos_token_id, params.use_cache, sampler
            )
        return output_ids


def generate_autoregress_encoder_decoder(
    model, input_ids, prefill_size, max_tokens, eos_token_id
):
    cur_seq_size = 1

    output_ids = input_ids
    while cur_seq_size < max_tokens:
        # logits = model.forward(output_ids)
        # if output_ids.shape[0] < model.embedding_decoder.activations_input[0].shape[0]:
        #     output_ids = F.pad(output_ids, (model.embedding_decoder.activations_input[0].shape[0] - output_ids.shape[0], 0))
        
        print("TYPES: ", type(output_ids), type(model.embedding_decoder.activations_input[0]))
        print("SHAPES: ", output_ids.shape, model.embedding_decoder.activations_input[0].shape)
        nntile.functions.copy_async(output_ids, model.embedding_decoder.activations_input[0])
        model.embedding_decoder.forward_async()
        model.transformer.decoder.forward_async()
        model.lm_head.forward_async()
        logits = model.lm_head.activations_output[0].value

        # TODO: add starpu function for argmax
        logits_np = nnt_constructors.to_numpy(logits)
        pred_token = np.argmax(logits_np[:, cur_seq_size - 1, :])

        if pred_token == eos_token_id:
            return output_ids, cur_seq_size

        # TODO: add starpu function for scalar assign
        output_ids_np = nnt_constructors.to_numpy(output_ids)
        output_ids_np[cur_seq_size, 0] = pred_token
        output_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids, cur_seq_size


def generate_autoregress(
    model, input_ids, prefill_size, max_tokens, eos_token_id
):
    cur_seq_size = prefill_size

    output_ids = input_ids
    while cur_seq_size < max_tokens:
        # logits = model.forward(output_ids)
        nntile.functions.copy_async(output_ids, model.embedding_decoder.activations_input[0].value)
        model.embedding_decoder.forward_async()
        model.forward_async()
        logits = model.lm_head.activations_output[0].value

        # TODO: add starpu function for argmax
        logits_np = nnt_constructors.to_numpy(logits)
        pred_token = np.argmax(logits_np[:, cur_seq_size - 1, :])

        if pred_token == eos_token_id:
            return output_ids, cur_seq_size

        # TODO: add starpu function for scalar assign
        output_ids_np = nnt_constructors.to_numpy(output_ids)
        output_ids_np[cur_seq_size, 0] = pred_token
        output_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids, cur_seq_size


def generate_autoregress_dynamic(
    model, input_ids, max_tokens, eos_token_id, use_cache, sampler
):
    cur_seq_size = input_ids.shape[0]

    kv_caches = None
    if use_cache:
        kv_caches = KVCacheStorage()

    output_ids_np = nntc.to_numpy(input_ids)

    while cur_seq_size < max_tokens:
        logits_nnt, kv_caches = model.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=use_cache,
            kv_caches=kv_caches,
        )
        output_value_np = nntc.to_numpy(logits_nnt.value)

        # TODO: add starpu function for argmax
        pred_token = sampler.sample(output_value_np[:, -1, :])
        pred_token = pred_token[0, 0]
        if pred_token == eos_token_id:
            return nntc.from_array(output_ids_np), cur_seq_size

        # TODO: add starpu function for concatenation
        output_ids_np = np.concatenate(
            [output_ids_np, pred_token[None, None]], axis=0
        )
        if use_cache:
            input_ids = nntc.from_array(
                pred_token[None, None].astype(np.int64)
            )
        else:
            input_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return nntc.from_array(output_ids_np), cur_seq_size


async def generate_autoregress_dynamic_async(
    model, input_ids, max_tokens, eos_token_id, use_cache, sampler
):
    cur_seq_size = input_ids.shape[0]

    kv_caches = None
    if use_cache:
        kv_caches = KVCacheStorage()

    output_ids_np = await nntc.to_numpy_async(input_ids)

    while cur_seq_size < max_tokens:
        logits_nnt, kv_caches = model.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=use_cache,
            kv_caches=kv_caches,
        )
        output_value_np = await nntc.to_numpy_async(logits_nnt.value)

        # TODO: add starpu function for argmax
        pred_token = sampler.sample(output_value_np[:, -1, :])
        pred_token = pred_token[0, 0]
        if pred_token == eos_token_id:
            return nntc.from_array(output_ids_np), cur_seq_size

        # TODO: add starpu function for concatenation
        output_ids_np = np.concatenate(
            [output_ids_np, pred_token[None, None]], axis=0
        )
        if use_cache:
            input_ids = nntc.from_array(
                pred_token[None, None].astype(np.int64)
            )
        else:
            input_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return nntc.from_array(output_ids_np), cur_seq_size
