/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/layer/gelutanh.cc
 * Approximate GeLU layer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-01
 * */

#include "nntile/layer/gelutanh.hh"

namespace nntile
{
namespace layer
{

// Explicit instantiations
template
class GeLUTanh<fp32_t>;

template
class GeLUTanh<fp64_t>;

} // namespace layer
} // namespace nntile


