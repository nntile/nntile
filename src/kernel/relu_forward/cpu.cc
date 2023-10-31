/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu_forward/cpu.cc
 * Forward ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-14
 * */

#include "nntile/kernel/relu_forward/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace relu_forward
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Forward ReLU operation on CPU
/*! Does the following per-element operation:
 * dst[i] = max(src[i], 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input array
 * @params[out] dst: Output array
 * */
{
    constexpr T zero = 0;
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = std::fmax(src[i], zero);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace relu_forward
} // namespace kernel
} // namespace nntile

