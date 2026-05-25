/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2.hh
 * Convenience header for GPT-2 model components.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/model/gpt2/gpt2_attention.hh>
#include <nntile/graph/model/gpt2/gpt2_block.hh>
#include <nntile/graph/model/gpt2/gpt2_causal.hh>
#include <nntile/graph/model/gpt2/gpt2_config.hh>
#include <nntile/graph/model/gpt2/gpt2_model.hh>
#include <nntile/graph/model/gpt2/gpt2_mlp.hh>
