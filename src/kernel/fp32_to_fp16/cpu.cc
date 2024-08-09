/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fp32_to_fp16/cpu.cc
 * Convert fp32_t array into fp16_t array on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fp32_to_fp16/cpu.hh"
#include <cuda_fp16.h>

namespace nntile::kernel::fp32_to_fp16
{

void cpu(Index nelems, const fp32_t *src_, fp16_t *dst_)
    noexcept
/*!
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input array
 * @params[out] dst: Output array
 * */
{
    auto src = reinterpret_cast<const float *>(src_);
    auto dst = reinterpret_cast<__half *>(dst_);
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = __float2half(src[i]);
    }
}

} // namespace nntile::kernel::fp32_to_fp16
