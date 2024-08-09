/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fp16_to_fp32/cuda.cu
 * Convert fp16_t array into fp32_t array on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/fp16_to_fp32/cuda.hh"
#include <cuda_fp16.h>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::fp16_to_fp32
{

static __global__
void cuda_kernel(Index nelems, const __half *src, float *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = __half2float(src[i]);
    }
}

void cuda(cudaStream_t stream, Index nelems, const fp16_t *src_, fp32_t *dst_)
    noexcept
/*!
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input array
 * @params[out] dst: Output array
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    const __half *src = reinterpret_cast<const __half *>(src_);
    float *dst = reinterpret_cast<float *>(dst_);
    (cuda_kernel)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

} // namespace nntile::kernel::fp16_to_fp32
