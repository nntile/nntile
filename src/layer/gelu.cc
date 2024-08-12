/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/gelu.cc
 * GeLU layer
 *
 * @version 1.1.0
 * */

#include "nntile/layer/gelu.hh"

namespace nntile::layer
{

// Explicit instantiations
template
class GeLU<fp32_t>;

template
class GeLU<fp64_t>;

} // namespace nntile::layer
