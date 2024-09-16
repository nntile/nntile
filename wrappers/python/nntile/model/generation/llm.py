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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

import nntile
import nntile.utils.constructors as nntc
from nntile.tensor import Tensor
from nntile.utils import constructors as nnt_constructors


class GenerationMode(Enum):
    Greedy = "Greedy"
    TopK = "TopK"
    TopP = "TopP"


@dataclass
class GenerationParams:
    max_tokens: int
    use_cache: bool = True
    need_static_padding: bool = False
    top_k: int | None = None
    top_p_thr: float | None = None
    temperature: float = 1


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
                raise Exception(
                    "No support for kvcache for static inference"
                )
            output_ids = generate_autoregress(
                self, input_ids, prefill_size,
                self.eos_token_id, params, sampler
            )
        else:
            output_ids = generate_autoregress_dynamic(
                self, input_ids, self.eos_token_id, params, sampler
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
        if params.need_static_padding:
            raise Exception(
                "No support for async static inference"
            )
        else:
            output_ids = await generate_autoregress_dynamic_async(
                self, input_ids, self.eos_token_id, params, sampler
            )

        return output_ids


def generate_autoregress(model, input_ids, prefill_size, eos_token_id, params):
    cur_seq_size = prefill_size

    output_ids = input_ids
    while cur_seq_size < params.max_tokens:
        logits = model.forward(output_ids)

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
    model, input_ids, eos_token_id, params, sampler
):
    cur_seq_size = input_ids.shape[0]

    kv_caches = None

    output_ids_np = nntc.to_numpy(input_ids)

    while cur_seq_size < params.max_tokens:
        logits_nnt, kv_caches = model.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=params.use_cache, kv_caches=kv_caches
        )
        output_value_np = nntc.to_numpy(logits_nnt.value)

        # TODO: add starpu function for argmax
        pred_token = sampler.sample(output_value_np[:, -1, :])
        if pred_token == eos_token_id:
            return nntc.from_array(output_ids_np), cur_seq_size

        # TODO: add starpu function for concatenation
        output_ids_np = np.concatenate(
            [output_ids_np, pred_token[None, None]], axis=0
        )
        if params.use_cache:
            input_ids = nntc.from_array(
                pred_token[None, None].astype(np.int64)
            )
        else:
            input_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return nntc.from_array(output_ids_np), cur_seq_size


async def generate_autoregress_dynamic_async(
        model, input_ids, eos_token_id, params, sampler
):
    cur_seq_size = input_ids.shape[0]

    kv_caches = None

    output_ids_np = await nntc.to_numpy_async(input_ids)

    while cur_seq_size < params.max_tokens:
        logits_nnt, kv_caches = model.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=params.use_cache, kv_caches=kv_caches
        )
        output_value_np = await nntc.to_numpy_async(logits_nnt.value)

        # TODO: add starpu function for argmax
        pred_token = sampler.sample(output_value_np[:, -1, :])
        if pred_token == eos_token_id:
            return nntc.from_array(output_ids_np), cur_seq_size

        # TODO: add starpu function for concatenation
        output_ids_np = np.concatenate(
            [output_ids_np, pred_token[None, None]], axis=0
        )
        if params.use_cache:
            input_ids = nntc.from_array(
                pred_token[None, None].astype(np.int64)
            )
        else:
            input_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return nntc.from_array(output_ids_np), cur_seq_size


def softmax_fn(arr):
    arr_exp = np.e**arr
    return arr_exp / arr_exp.sum()


def sample_topk(logits, k, temperature):
    logits = logits[:, 0]
    argsorted_logits = np.argsort(logits)
    top_indices = argsorted_logits[-k:]
    top_probas = softmax_fn(logits[top_indices] / temperature)
    next_token = np.random.default_rng().choice(top_indices, p=top_probas)
    return next_token


def sample_topp(logits, p_thr, temperature):
    logits = logits[:, 0]
    argsorted_logits = np.argsort(logits)
    sorted_probas = softmax_fn(logits[argsorted_logits])
    cumsum_from_largest = np.cumsum(sorted_probas[::-1])

    topp_k = np.searchsorted(cumsum_from_largest, p_thr)
    top_indices = argsorted_logits[-topp_k:]
    top_probas = softmax_fn(logits[top_indices] / temperature)
    next_token = np.random.default_rng().choice(top_indices, p=top_probas)
    return next_token


def sample_greedy(logits):
    logits = logits[:, 0]
    next_token = np.argmax(logits)
    return next_token


class BaseSampler(ABC):
    @abstractmethod
    def sample(logits):
        pass


class GreedySampler(BaseSampler):
    def sample(self, logits):
        return sample_greedy(logits)


class TopKSampler(BaseSampler):
    def __init__(self, k, temperature):
        self.k = k
        self.temperature = temperature

    def sample(self, logits):
        return sample_topk(logits, self.k, self.temperature)


class TopPSampler(BaseSampler):
    def __init__(self, p_thr, temperature):
        self.p_thr = p_thr
        self.temperature = temperature

    def sample(self, logits):
        return sample_topp(logits, self.p_thr, self.temperature)


def get_sampler(mode, params):
    if mode == GenerationMode.Greedy:
        return GreedySampler()
    elif mode == GenerationMode.TopK:
        return TopKSampler(params.top_k, params.temperature)
    elif mode == GenerationMode.TopP:
        return TopPSampler(params.top_p_thr, params.temperature)
    else:
        raise Exception("Unknown sampler")
