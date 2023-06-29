/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sqrt/cuda.cu
 * Per-element sqrt of buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-29
 * */

#include "nntile/kernel/sqrt/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace sqrt
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = ::sqrt(dst[i]);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *dst)
    noexcept
//! Per-element sqrt of buffer
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *dst)
    noexcept;

} // namespace sqrt
} // namespace kernel
} // namespace nntile

