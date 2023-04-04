/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu_backward/cpu.cc
 * Backward ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-04
 * */

#include "nntile/kernel/relu_backward/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace relu_backward
{

template<typename T>
void cpu(Index nelems, const T *x, const T *dy, T *dx)
    noexcept
//! Backward ReLU operation on CPU
/*! Does the following per-element operation:
 * backward_ReLU(x, dy) = dy if x > 0 and 0 otherwise
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward ReLU
 * @params[in] dy: Gradient over output of forward ReLU
 * @params[out] dx: Gradient over input of forward ReLU
 * */
{
    constexpr T zero = 0;;
    for(Index i = 0; i < nelems; ++i)
    {
        if(x[i] > zero)
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = zero;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *x, const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *x, const fp64_t *dy, fp64_t *dx)
    noexcept;

} // namespace relu_backward
} // namespace kernel
} // namespace nntile

