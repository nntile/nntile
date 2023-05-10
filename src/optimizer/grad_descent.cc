/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/optimizer/grad_descent.cc
 * Gradient descent with constant learning rate
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#include "nntile/optimizer/grad_descent.hh"

namespace nntile
{
namespace optimizer
{

// Explicit instantiation
template
class GradDescent<fp32_t>;

template
class GradDescent<fp64_t>;

} // namespace optimizer
} // namespace nntile

