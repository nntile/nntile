/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/optimizer/sgd.cc
 * SGD with constant learning rate, momentum, nesterov regime and weight decay
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/optimizer/sgd.hh"

namespace nntile
{
namespace optimizer
{

// Explicit instantiation
template
class SGD<fp32_t>;

template
class SGD<fp64_t>;

} // namespace optimizer
} // namespace nntile

