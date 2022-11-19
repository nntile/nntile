/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/model/deep_linear.cc
 * Deep linear network
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-11-18
 * */

#include "nntile/model/deep_linear.hh"

namespace nntile
{
namespace model
{

// Explicit instantiation
template
class DeepLinear<fp32_t>;

template
class DeepLinear<fp64_t>;

} // namespace model
} // namespace nntile


