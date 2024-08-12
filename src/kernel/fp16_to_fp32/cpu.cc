/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fp16_to_fp32/cpu.cc
 * Convert fp16_t array into fp32_t array on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fp16_to_fp32/cpu.hh"
#include <cuda_fp16.h>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::fp16_to_fp32
{

void cpu(Index nelems, const fp16_t *src_, fp32_t *dst_)
    noexcept
/*!
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input array
 * @params[out] dst_: Output array
 * */
{
    auto src = reinterpret_cast<const __half *>(src_);
    auto dst = reinterpret_cast<float *>(dst_);
    for(Index i = 0; i < nelems; ++i)
    {
        dst[i] = __half2float(src[i]);
    }
}

} // namespace nntile::kernel::fp16_to_fp32
