/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cuda.cu
 * Fill operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#include "nntile/kernel/mask_scalar/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace mask_scalar
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, bool_t* mask, T val, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        if (!mask[i])
        {
            data[i] = val;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, bool_t* mask,  T val, T *data)
    noexcept
//! Mask operation with given value on CUDA
/*! Seta all elements to the provided value if mask value is 0
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] mask: mask buffer
 * @param[in] val: Input value
 * @params[in,out] data: Output buffer
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, mask, val, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, bool_t* mask, fp32_t val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, bool_t* mask, fp64_t val, fp64_t *data)
    noexcept;

} // namespace mask_scalar
} // namespace kernel
} // namespace nntile