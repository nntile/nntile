/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/addcdiv/cuda.cu
 * Addcdiv operation for buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/addcdiv/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace addcdiv
{

template<typename T>
static __global__
void cuda_kernel(T val, T eps, Index nelems, const T *nom, const T* denom,
        T *res)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        res[i] += val * nom[i] / (denom[i] + eps);
    }
}

template<typename T>
void cuda(cudaStream_t stream, T val, T eps, Index nelems, const T *nom,
        const T* denom, T *res)
    noexcept
//! Addcdiv operation of buffers
/*! One of the buffers serves as output
 *
 * @param[in] val: scalar multiplicator
 * @param[in] eps: small value to avoid division by zero
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] nom: buffer to store the elements from nominator of ratio
 * @param[in] denom: buffer to store the elements from denominator of ratio
 * @param[inout] res: Input buffers that contains output in the end
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(val, eps, nelems, nom,
            denom, res);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, fp32_t val, fp32_t eps, Index nelems,
        const fp32_t *nom, const fp32_t* denom, fp32_t *res)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, fp64_t val, fp64_t eps, Index nelems,
        const fp64_t *nom, const fp64_t* denom, fp64_t *res)
    noexcept;

} // namespace addcdiv
} // namespace kernel
} // namespace nntile

