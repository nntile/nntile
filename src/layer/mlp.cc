/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/mlp.cc
 * Multilayer perceptron
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-01
 * */

#include <nntile/layer/mlp.hh>

namespace nntile
{
namespace layer
{

// Explicit instantiations
template
class MLP<fp32_t>;

template
class MLP<fp64_t>;

} // namespace layer
} // namespace nntile

