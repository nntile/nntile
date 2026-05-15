/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_infnan/cpu.cc
 * Test whether at least one element of a buffer on CPU is NaN or Inf
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_infnan/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::accumulate_infnan
{

template<typename T>
void cpu(Index nelems, const T *src_, Index *dst)
    noexcept
//! Test the given array on NaN of Inf values
/*! For a provided input array of nelems elements test
 * if at least any of the element is NaN or Inf.
 *
 * @param[in] nelems: Number of elements in the input array
 * @param[in] src_: Input contiguous array of nelems elements
 * @param[inout] dst: Output value true if at least one element is
 * NaN or Inf
 * */
{
    Index current_flag = 0;
    using Y = typename T::repr_t;
    Y src_val = 0;
    for(Index i = 0; i < nelems; ++i)
    {
        src_val = static_cast<Y>(src_[i]);
        if(std::isnan(src_val) || std::isinf(src_val))
        {
            current_flag = 1;
            break;
        }
    }
    // Update output value
    if((current_flag == 1) && (dst[0] == 0))
    {
        dst[0] = current_flag;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src,
        Index *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src,
        Index *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *src,
        Index *dst)
    noexcept;

template
void cpu<fp16_t>(Index nelems, const fp16_t *src,
        Index *dst)
    noexcept;

} // namespace nntile::kernel::accumulate_infnan
