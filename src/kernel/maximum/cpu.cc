/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maximum/cpu.cc
 * Per-element maximum of two buffers on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-11-06
 * */

#include "nntile/kernel/maximum/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace maximum
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Per-element maximum of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] src: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    // Cycle over buffers
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = std::fmax(src[i], dst[i]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

} // namespace maximum
} // namespace kernel
} // namespace nntile
