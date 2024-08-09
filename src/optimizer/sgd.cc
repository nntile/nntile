/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/optimizer/sgd.cc
 * SGD with constant learning rate, momentum, nesterov regime and weight decay
 *
 * @version 1.1.0
 * */

#include "nntile/optimizer/sgd.hh"

namespace nntile::optimizer
{

// Explicit instantiation
template
class SGD<fp32_t>;

template
class SGD<fp64_t>;

} // namespace nntile::optimizer
