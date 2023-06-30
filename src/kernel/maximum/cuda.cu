/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/pow/cuda.cu
 * Power operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/maximum/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace maximum
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T* src, T* dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = ::max(dst[i], src[i]);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T* src, T* dst)
    noexcept
//! Inplace maximum operation on CUDA
/*! Does the following per-element operation:
 * dst[i] := max(src[i], dst[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: input buffer 
 * @params[inout] dst: buffer for comparison and store maximum
 * */
{
    dim3 blocks(256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t* src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t* src, fp64_t *dst)
    noexcept;

} // namespace maximum
} // namespace kernel
} // namespace nntile

