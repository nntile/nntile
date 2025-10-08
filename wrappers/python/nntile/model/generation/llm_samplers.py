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
from scipy.special import softmax  # type: ignore[import-untyped]

from nntile.model.generation.llm_params import GenerationMode


def sample_topk(logits, k, temperature, num_samples=1, with_replace=False):
    assert len(logits.shape) <= 2, "Support logits or batch of logits only"
    batch_extended = logits if len(logits.shape) > 1 else logits[:, None]

    sampled = np.empty((batch_extended.shape[1], num_samples), dtype=np.int64)
    for batch_indx in range(batch_extended.shape[1]):
        single_logits = batch_extended[:, batch_indx]
        argsorted_logits = np.argsort(single_logits)
        top_indices = argsorted_logits[-k:]
        top_probas = softmax(single_logits[top_indices] / temperature, axis=0)
        indexes = np.random.default_rng().choice(
            top_indices, size=num_samples, p=top_probas, replace=with_replace
        )

        # sort sampled tokens according to their logits
        vals = single_logits[indexes]
        sorted_order = vals.argsort()
        indexes_sorted = indexes[sorted_order]

        sampled[batch_indx] = indexes_sorted
    return sampled


def sample_topp(logits, p_thr, temperature, num_samples=1, with_replace=False):
    assert len(logits.shape) <= 2, "Support logits or batch of logits only"
    batch_extended = logits if len(logits.shape) > 1 else logits[:, None]

    sampled = np.empty((batch_extended.shape[1], num_samples), dtype=np.int64)
    for batch_indx in range(batch_extended.shape[1]):
        single_logits = batch_extended[:, batch_indx]
        argsorted_logits = np.argsort(single_logits)
        sorted_probas = softmax(
            single_logits[argsorted_logits] / temperature, axis=0
        )
        cumsum_from_largest = np.cumsum(sorted_probas[::-1])

        topp_k = np.searchsorted(cumsum_from_largest, p_thr)
        top_indices = argsorted_logits[-topp_k:]
        top_probas = softmax(single_logits[top_indices] / temperature, axis=0)
        indexes = np.random.default_rng().choice(
            top_indices, size=num_samples, p=top_probas, replace=with_replace
        )

        # sort sampled tokens according to their logits
        vals = single_logits[indexes]
        sorted_order = vals.argsort()
        indexes_sorted = indexes[sorted_order]

        sampled[batch_indx] = indexes_sorted
    return sampled


def sample_greedy(logits, num_samples=1):
    assert len(logits.shape) <= 2, "Support logits or batch of logits only"
    batch_extended = logits if len(logits.shape) > 1 else logits[:, None]

    sampled = np.empty((batch_extended.shape[1], num_samples), dtype=np.int64)
    for batch_indx in range(batch_extended.shape[1]):
        single_logits = batch_extended[:, batch_indx]
        indexes = np.argpartition(single_logits, -num_samples)[-num_samples:]
        vals = single_logits[indexes]
        sorted_order = vals.argsort()
        indexes_sorted = indexes[sorted_order]

        sampled[batch_indx] = indexes_sorted

    return sampled


class BaseSampler(ABC):
    @abstractmethod
    def sample(logits):
        pass


class GreedySampler(BaseSampler):
    def sample(self, logits, num_samples=1):
        return sample_greedy(logits, num_samples=num_samples)


class TopKSampler(BaseSampler):
    def __init__(self, k, temperature):
        self.k = k
        self.temperature = temperature

    def sample(self, logits, num_samples=1):
        return sample_topk(
            logits, self.k, self.temperature, num_samples=num_samples
        )


class TopPSampler(BaseSampler):
    def __init__(self, p_thr, temperature):
        self.p_thr = p_thr
        self.temperature = temperature

    def sample(self, logits, num_samples=1):
        return sample_topp(
            logits, self.p_thr, self.temperature, num_samples=num_samples
        )


def get_sampler(mode, params):
    if mode == GenerationMode.Greedy:
        return GreedySampler()
    elif mode == GenerationMode.TopK:
        return TopKSampler(params.top_k, params.temperature)
    elif mode == GenerationMode.TopP:
        return TopPSampler(params.top_p_thr, params.temperature)
    else:
        raise Exception("Unknown sampler")
