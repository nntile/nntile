/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama.hh
 * Llama - alias for LlamaModel (base model without LM head).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/model/llama/llama_model.hh>

namespace nntile::model::llama
{

//! Llama - same as LlamaModel (embedding + decoder layers + final norm)
using Llama = LlamaModel;

} // namespace nntile::model::llama
