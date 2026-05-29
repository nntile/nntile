/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/llama/llama_causal_mask.hh
 * Deprecated include: use ``nntile/nn_graph/ops/sdpa_causal_mask.hh``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/nn_graph/ops/sdpa_causal_mask.hh>

namespace nntile::model::llama
{

using nntile::sdpa_causal_mask_bool_fortran_fill;

} // namespace nntile::model::llama
