/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/gelu.cc
 * GeLU layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-24
 * */

#include "nntile/layer/gelu.hh"

namespace nntile
{
namespace layer
{

// Explicit instantiations
template
class GeLU<fp32_t>;

template
class GeLU<fp64_t>;

} // namespace layer
} // namespace nntile


