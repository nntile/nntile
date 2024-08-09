/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fp32_to_fp16/cuda.cu
 * Convert fp32_t array into fp16_t array on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fp32_to_fp16/cuda.hh"
#include <cuda_fp16.h>

namespace nntile::kernel::fp32_to_fp16
{

static __global__
void cuda_kernel(Index nelems, const float *src, __half *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = __float2half(src[i]);
    }
}

void cuda(cudaStream_t stream, Index nelems, const fp32_t *src_, fp16_t *dst_)
    noexcept
/*!
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input array
 * @params[out] dst: Output array
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    const float *src = reinterpret_cast<const float *>(src_);
    __half *dst = reinterpret_cast<__half *>(dst_);
    (cuda_kernel)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

} // namespace nntile::kernel::fp32_to_fp16
