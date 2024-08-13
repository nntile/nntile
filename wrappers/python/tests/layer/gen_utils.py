# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/tests/layer/gen_utils.py
# Utils for testing autoregressive generation
#
# @version 1.1.0

import numpy as np

import nntile
import nntile.utils.constructors as nntc


def generate_greedy_logits_padding(
    attn_layer, input_ids, prefill_size, max_tokens
):
    cur_seq_size = prefill_size

    output_ids = input_ids
    while cur_seq_size < max_tokens:
        output_ids_np = nntc.to_numpy(output_ids)

        logits = attn_layer.forward_dynamic(
            nntile.tensor.TensorMoments(output_ids, None, False),
            use_cache=False,
        )
        logits_np = nntc.to_numpy(logits.value)

        output_ids_np = np.concatenate(
            [output_ids_np, logits_np[:, -1, :][:, None, :]], axis=1
        )
        output_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids


def generate_greedy_logits_dynamic_kvcache(
    attn_layer, input_ids, prefill_size, max_tokens
):
    cur_seq_size = prefill_size

    output_ids = input_ids

    is_prefill = True

    while cur_seq_size < max_tokens:
        output_ids_np = nntc.to_numpy(output_ids)

        logits = attn_layer.forward_dynamic(
            nntile.tensor.TensorMoments(input_ids, None, False),
            use_cache=(not is_prefill),
        )
        if is_prefill:
            is_prefill = False

        logits_np = nntc.to_numpy(logits.value)

        input_ids_np = logits_np[:, -1, :][:, None, :]
        output_ids_np = np.concatenate([output_ids_np, input_ids_np], axis=1)

        input_ids = nntc.from_array(input_ids_np)
        output_ids = nntc.from_array(output_ids_np)
        cur_seq_size += 1

    return output_ids
