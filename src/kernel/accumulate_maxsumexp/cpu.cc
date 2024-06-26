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
 * @version 1.0.0
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
    using Y = typename CPUComputeType<T>::value;
    constexpr Y zero{0.0};
    auto *src = reinterpret_cast<const Y *>(src_);
    auto *dst = reinterpret_cast<Y *>(dst_);
    for(Index i = 0; i < nelems; ++i)
    {
        // Do nothing if sum of exponents of source is zero
        if(src[2*i+1] != zero)
        {
            // Overwrite if old value of sum is zero
            if(dst[2*i+1] == zero)
            {
                dst[2*i] = src[2*i];
                dst[2*i+1] = src[2*i+1];
            }
            // Otherwise update based on maximum
            else if(dst[2*i] < src[2*i])
            {
                dst[2*i+1] = src[2*i+1] + dst[2*i+1]*std::exp(dst[2*i]-src[2*i]);
                dst[2*i] = src[2*i];
            }
            else
            {
                dst[2*i+1] += src[2*i+1]*std::exp(src[2*i]-dst[2*i]);
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

} // namespace nntile::kernel::accumulate_maxsumexp
