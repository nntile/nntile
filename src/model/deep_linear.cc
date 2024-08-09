/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/deep_linear.cc
 * Deep linear network
 *
 * @version 1.1.0
 * */

#include "nntile/model/deep_linear.hh"

namespace nntile::model
{

// Explicit instantiation
template
class DeepLinear<fp32_t>;

template
class DeepLinear<fp64_t>;

} // namespace nntile::model
