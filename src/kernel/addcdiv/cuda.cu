/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/addcdiv/cuda.cu
 * Addcdiv operation for buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/addcdiv/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::addcdiv
{

template<typename T>
static __global__
void cuda_kernel(Scalar val, Scalar eps, Index nelems, const T *nom, const T* denom,
        T *res)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < nelems)
    {
        Y res_val{res[i]};
        res[i] = res_val + (Y{val} * Y{nom[i]}) / (Y{denom[i]} + Y{eps});
    }
}

template<typename T>
void cuda(cudaStream_t stream, Scalar val, Scalar eps, Index nelems,
        const T *nom, const T *denom, T *res)
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
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(val, eps, nelems,
            nom, denom, res);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Scalar val, Scalar eps, Index nelems,
        const fp32_t *nom, const fp32_t *denom, fp32_t *res)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Scalar val, Scalar eps, Index nelems,
        const fp64_t *nom, const fp64_t *denom, fp64_t *res)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Scalar val, Scalar eps, Index nelems,
        const bf16_t *nom, const bf16_t *denom, bf16_t *res)
    noexcept;

} // namespace nntile::kernel::addcdiv
