/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cuda.cu
 * Logsumexp after computed maxsumexp result of a buffer on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/logsumexp/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::logsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T * __restrict__ maxsumexp_,
        T * __restrict__ logsumexp_)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::compat_t;
    using Z = typename CUDAComputeType<T>::value;
    const Z* maxsumexp = reinterpret_cast<const Z *>(maxsumexp_);
    Z* logsumexp = reinterpret_cast<Z *>(logsumexp_);
    Y maxsumexp_even = Y{maxsumexp[2*i]};
    Y maxsumexp_odd = Y{maxsumexp[2*i+1]};
    if(i < nelems)
    {
        logsumexp[i] = Z{maxsumexp_even + ::log(maxsumexp_odd)};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *maxsumexp_,
        T *logsumexp_)
    noexcept
//! Logsumexp of buffer
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] maxsumexp_: Input buffer, result of maxsumexp operation 
 * @param[out] logsumexp_: Output buffers that contains output in the end
 * */
{
    dim3 blocks((nelems+31)/32), threads(32);
    // using Y = typename CUDAComputeType<T>::value;
    // auto maxsumexp = reinterpret_cast<const Y *>(maxsumexp_);
    // auto logsumexp = reinterpret_cast<Y *>(logsumexp_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, maxsumexp_,
            logsumexp_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *maxsumexp,
        fp32_t *logsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *maxsumexp,
        fp64_t *logsumexp)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *maxsumexp,
        bf16_t *logsumexp)
    noexcept;

} // namespace nntile::kernel::logsumexp
