/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/linear.cc
 * Fully connected dense linear layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-20
 * */

#include "nntile/layer/linear.hh"

namespace nntile
{
namespace layer
{

// Explicit instantiation
template
class Linear<fp32_t>;

template
class Linear<fp64_t>;

} // namespace layer
} // namespace nntile


