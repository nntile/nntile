/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/isfinite/cuda.cu
 * Accumulate flags for NaN and Inf in a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/isfinite/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::isfinite
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, bool_t *dst)
{

    if(bool_t::repr_t{dst[0]} == 1)
    {
        return;
    }

    using Y = typename T::repr_t;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nelems)
    {
        Y val = Y{src[idx]};
        // Check for NaN or Inf
        if(isnan(val) || isinf(val))
        {
            dst[0] = 1;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, bool_t *dst)
    noexcept
//! Accumulate flags for Inf and NaN elements in buffer
/*! For a provided input array of nelems elements indicate
 *  whether there is NaN of Inf elements

 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Number of elements in the input array
 * @param[in] src: Input contiguous array
 * @param[inout] dst: Output scalar (single element array)
 * */
{
    // Use a single block with up to 1024 threads
    dim3 threads(1024);
    dim3 blocks(1);

    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t *src, bool_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t *src, bool_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t *src, bool_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t *src, bool_t *dst)
    noexcept;

} // namespace nntile::kernel::isfinite
