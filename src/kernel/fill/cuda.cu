/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cuda.cu
 * Fill operation on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/fill/cuda.hh"

namespace nntile::kernel::fill
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T val, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        data[i] = val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T val, T *data)
    noexcept
//! Fill operation on CUDA
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val: Input value
 * @params[out] data: Output buffer
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, val, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t val, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::fill
