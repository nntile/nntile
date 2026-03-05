/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama.hh
 * Convenience header for Llama model components.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/model/llama/llama_config.hh>
#include <nntile/model/llama/llama_mlp.hh>
#include <nntile/model/llama/llama_attention.hh>
#include <nntile/model/llama/llama_decoder.hh>
#include <nntile/model/llama/llama_model.hh>
#include <nntile/model/llama/llama.hh>
#include <nntile/model/llama/llama_causal.hh>
