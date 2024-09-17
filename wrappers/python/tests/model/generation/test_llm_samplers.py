# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/model/generation/test_llm_samplers.py
# Test for implementation of Deep ReLU model in NNTile
#
# @version 1.1.0

import numpy as np
import pytest
import scipy

# import nntile.utils.constructors as nntc
from nntile.model.generation.llm_params import GenerationMode, GenerationParams
from nntile.model.generation.llm_samplers import (
    TopKSampler, TopPSampler, get_sampler)


@pytest.mark.parametrize("embed_size,batch_size", [[100, 7]])
def test_greedy(numpy_rng, embed_size, batch_size):
    mode = GenerationMode.Greedy
    params = GenerationParams(0)
    logits = numpy_rng.random([embed_size, batch_size])
    sampler = get_sampler(mode, params)

    found_tokens = sampler.sample(logits)
    expected_tokens = np.argmax(logits, axis=0)

    assert (found_tokens == expected_tokens).all(), \
    f"{found_tokens} != {expected_tokens}"


@pytest.mark.parametrize("n_resamples", [20])
def test_topk(n_resamples):
    inp = np.array([[0.39437038, 0.686569],
       [0.07847447, 0.92659984],
       [0.88788324, 0.06391566],
       [0.07849913, 0.84102763],
       [0.77358009, 0.68963694]]
    )

    expected0 = np.argsort(inp[:, 0])
    expected1 = np.argsort(inp[:, 1])

    temperature = 1
    for k in range(1, inp.shape[0]):
        sampled = []

        for _ in range(n_resamples):
            sampler = TopKSampler(k, temperature)
            tokens = sampler.sample(inp)
            sampled.append(tokens)

        sampled = np.vstack(sampled)
        unique0 = np.unique(sampled[:, 0])
        unique1 = np.unique(sampled[:, 1])

        assert unique0.sort() == expected0[-k:].sort()
        assert unique1.sort() == expected1[-k:].sort()


@pytest.mark.parametrize("n_resamples", [20])
def test_topp(n_resamples):
    inp = np.array([[0.39437038, 0.686569],
       [0.07847447, 0.92659984],
       [0.88788324, 0.06391566],
       [0.07849913, 0.84102763],
       [0.77358009, 0.68963694]]
    )

    sfi = scipy.special.softmax(inp, axis=0)
    thrs0 = np.cumsum(np.sort(sfi[:, 0])[::-1])
    thrs1 = np.cumsum(np.sort(sfi[:, 1])[::-1])

    expected0 = np.argsort(inp[:, 0])
    expected1 = np.argsort(inp[:, 1])

    thresholds = [0.3, 0.5, 0.7, 0.9]

    temperature = 1
    for p_thr in thresholds:
        sampled = []

        for _ in range(n_resamples):
            sampler = TopPSampler(p_thr, temperature)
            tokens = sampler.sample(inp)
            sampled.append(tokens)

        sampled = np.vstack(sampled)
        unique0 = np.unique(sampled[:, 0])
        unique1 = np.unique(sampled[:, 1])

        k0 = np.searchsorted(thrs0, p_thr)
        k1 = np.searchsorted(thrs1, p_thr)

        assert unique0.sort() == expected0[-k0:].sort()
        assert unique1.sort() == expected1[-k1:].sort()
