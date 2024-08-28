/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_maxsumexp/cpu.cc
 * Accumulate maxsumexp buffers on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_maxsumexp/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::accumulate_maxsumexp
{

template<typename T>
void cpu(Index nelems, const T* src_, T* dst_)
    noexcept
//! Accumulate two maxsumexp buffers on CPU
/*! Performs the following operation:
 *      dst[2*i+1] = dst[2*i+1]*exp(dst[2*i]) + src[2*i+1]*exp(src[2*i]),
 *      dst[2*i] = max(src[2*i], dst[2*i]).
 *
 * @param[in] nelems: Number of (max,sumexp) pairs of the src and dst tensors
 * @param[in] src_: Source tensor
 * @param[inout] dst_: Destination of the maxsumexp accumulation
 * */
{
    constexpr typename T::repr_t zero{0.0};
    for(Index i = 0; i < nelems; ++i)
    {
        auto src_odd = static_cast<typename T::repr_t>(src_[2*i+1]);
        auto src_even = static_cast<typename T::repr_t>(src_[2*i]);
        auto dst_odd = static_cast<typename T::repr_t>(dst_[2*i+1]);
        auto dst_even = static_cast<typename T::repr_t>(dst_[2*i]);
        // Do nothing if sum of exponents of source is zero
        if(src_odd != zero)
        {
            // Overwrite if old value of sum is zero
            if(dst_odd == zero)
            {
                dst_[2*i] = src_[2*i];
                dst_[2*i+1] = src_[2*i+1];
            }
            // Otherwise update based on maximum
            else if(dst_even < src_even)
            {
                dst_[2*i+1] = static_cast<T>(src_odd + dst_odd*std::exp(dst_even - src_even));
                dst_[2*i] = src_[2*i];
            }
            else
            {
                dst_[2*i+1] = static_cast<T>(dst_odd + src_odd * std::exp(src_even - dst_even));
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t* src, fp32_t* dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t* src, fp64_t* dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t* src, bf16_t* dst)
    noexcept;

} // namespace nntile::kernel::accumulate_maxsumexp
