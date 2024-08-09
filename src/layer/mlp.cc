/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/mlp.cc
 * Multilayer perceptron
 *
 * @version 1.1.0
 * */

#include <nntile/layer/mlp.hh>

namespace nntile::layer
{

// Explicit instantiations
template
class MLP<fp32_t>;

template
class MLP<fp64_t>;

} // namespace nntile::layer
