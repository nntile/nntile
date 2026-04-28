/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox.hh
 * Alias for GptneoxModel (base model without LM head).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/model/gptneox/gptneox_model.hh>

namespace nntile::model::gptneox
{

//! Gptneox - same as GptneoxModel (embedding + decoder layers + final norm)
using Gptneox = GptneoxModel;

} // namespace nntile::model::gptneox
