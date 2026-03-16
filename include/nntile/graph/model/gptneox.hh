/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox.hh
 * Convenience header for GPT-NeoX model components.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/model/gptneox/gptneox_config.hh>
#include <nntile/graph/model/gptneox/gptneox_mlp.hh>
#include <nntile/graph/model/gptneox/gptneox_attention.hh>
#include <nntile/graph/model/gptneox/gptneox_decoder.hh>
#include <nntile/graph/model/gptneox/gptneox_model.hh>
#include <nntile/graph/model/gptneox/gptneox.hh>
#include <nntile/graph/model/gptneox/gptneox_causal.hh>
