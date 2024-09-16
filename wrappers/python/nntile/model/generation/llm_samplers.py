# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/generation/llm_samplers.py
#
# @version 1.1.0

from abc import ABC, abstractmethod

import numpy as np

from nntile.model.generation.llm_params import GenerationMode


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
