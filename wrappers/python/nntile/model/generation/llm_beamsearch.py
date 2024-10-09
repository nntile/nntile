# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/generation/llm_beamsearch.py
#
# @version 1.1.0

import numpy as np
import scipy

import nntile.utils.constructors as nntc
from nntile.layer.cache_utils import ParallelSamplingCacheStorage
from nntile.model.generation.llm_params import ParallelSamplingMode
from nntile.tensor import TensorMoments


def reduce_parallel(beams_logits, sampler):
    logits = np.vstack(beams_logits)
    indexes = sampler.sample(logits.T).flatten()
    beams = list(range(indexes.shape[0]))
    return indexes.tolist(), beams


def reduce_global(beams_logits, sampler):
    logits = np.vstack(beams_logits)
    flat_logits = logits.flatten()
    indexes_sorted = sampler.sample(
        flat_logits, num_samples=logits.shape[0]
    ).flatten()

    beams = [index // logits.shape[1] for index in indexes_sorted]
    indexes = [index % logits.shape[1] for index in indexes_sorted]
    return indexes, beams


def generate_parallel(
    model,
    input_ids,
    max_tokens,
    eos_token_id,
    num_beams,
    sampler,
    sampling_mode=ParallelSamplingMode.BeamSearch,
):
    assert input_ids.shape[1] == 1
    cur_seq_size = input_ids.shape[0]
    output_ids_np = nntc.to_numpy(input_ids)

    cache_storage = ParallelSamplingCacheStorage(num_beams)

    out_prefill = model.forward_dynamic(
        TensorMoments(input_ids, None, False), False, cache_storage
    )
    logits, _ = out_prefill
    logits_np = nntc.to_numpy(logits.value)
    last_logits = logits_np[:, -1, :]

    next_tokens = np.argpartition(last_logits[:, 0], -num_beams)[-num_beams:]

    beams_outs = [next_tokens]
    if sampling_mode == ParallelSamplingMode.BeamSearch:
        logits = [
            [scipy.special.softmax(last_logits) for _ in range(num_beams)]
        ]
    else:
        logits = [[last_logits for _ in range(num_beams)]]
    cur_seq_size += 1

    while cur_seq_size < max_tokens:
        beams_logits = []

        for beam, t in enumerate(next_tokens):
            beam_kv_caches = cache_storage.get_beam(beam)

            next_token = np.asfortranarray([t]).astype(np.int64)[:, None]
            next_token_nnt = nntc.from_array(next_token)

            out_decode = model.forward_dynamic(
                TensorMoments(next_token_nnt, None, False),
                True,
                beam_kv_caches,
            )

            logits_decode, _ = out_decode
            logits_decode_np = nntc.to_numpy(logits_decode.value)
            last_decode_logits = logits_decode_np[:, -1, :]
            beams_logits.append(last_decode_logits)
        if sampling_mode == ParallelSamplingMode.Parallel:
            tokens, beams_ids = reduce_parallel(
                [bl[:, 0] for bl in beams_logits], sampler=sampler
            )
        else:
            prev_logits = logits[-1]
            if sampling_mode == ParallelSamplingMode.BeamSearch:
                weighted_logits = [
                    scipy.special.softmax(bl[:, 0])
                    * prev_logits[beam].flatten()[
                        np.array(next_tokens).flatten()[beam]
                    ]
                    for beam, bl in enumerate(beams_logits)
                ]
                beams_logits = [el[:, None] for el in weighted_logits]
            else:
                weighted_logits = [
                    scipy.special.softmax(bl[:, 0])
                    * scipy.special.softmax(prev_logits[beam].flatten())[
                        np.array(next_tokens).flatten()[beam]
                    ]
                    for beam, bl in enumerate(beams_logits)
                ]

            tokens, beams_ids = reduce_global(weighted_logits, sampler=sampler)

        cache_storage.reduce(beams_ids)

        beams_outs = [np.array(b)[np.array(beams_ids)] for b in beams_outs]
        beams_logits_save = [beams_logits[beam] for beam in beams_ids]

        next_tokens = tokens
        beams_outs.append(next_tokens)
        logits.append(beams_logits_save)

        cur_seq_size += 1

        if tokens[-1] == eos_token_id:
            break

    result_sequences = np.concatenate(
        [
            output_ids_np.repeat(num_beams, axis=1),
        ]
        + [np.array(el)[None, :] for el in beams_outs]
    )

    return nntc.from_array(result_sequences), cur_seq_size
