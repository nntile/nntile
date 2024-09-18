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
from scipy.special import softmax

from nntile.model.generation.llm_params import GenerationMode


def sample_topk(logits, k, temperature):
    assert len(logits.shape) <= 2, "Support logits or batch of logits only"
    batch_extended = logits if len(logits.shape) > 1 else logits[:, None]

    sampled = np.empty((logits.shape[1],), dtype=np.int64)
    for batch_indx in range(logits.shape[1]):
        single_logits = batch_extended[:, batch_indx]
        argsorted_logits = np.argsort(single_logits)
        top_indices = argsorted_logits[-k:]
        top_probas = softmax(single_logits[top_indices] / temperature, axis=0)
        next_token = np.random.default_rng().choice(top_indices, p=top_probas)
        sampled[batch_indx] = next_token
    return sampled


def sample_topp(logits, p_thr, temperature):
    assert len(logits.shape) <= 2, "Support logits or batch of logits only"
    batch_extended = logits if len(logits.shape) > 1 else logits[:, None]

    sampled = np.empty((logits.shape[1],), dtype=np.int64)
    for batch_indx in range(logits.shape[1]):
        single_logits = batch_extended[:, batch_indx]
        argsorted_logits = np.argsort(single_logits)
        sorted_probas = softmax(
            single_logits[argsorted_logits] / temperature,
            axis=0
        )
        cumsum_from_largest = np.cumsum(sorted_probas[::-1])

        topp_k = np.searchsorted(cumsum_from_largest, p_thr)
        top_indices = argsorted_logits[-topp_k:]
        top_probas = softmax(single_logits[top_indices] / temperature, axis=0)
        next_token = np.random.default_rng().choice(top_indices, p=top_probas)
        sampled[batch_indx] = next_token
    return sampled


def sample_greedy(logits):
    next_token = np.argmax(logits, axis=0)
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
