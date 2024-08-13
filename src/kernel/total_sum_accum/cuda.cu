/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/total_sum_accum/cuda.cu
 * total_sum_accum operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/total_sum_accum/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::total_sum_accum
{

template<typename T>
static __global__
void cuda_kernel(Scalar alpha, Index n_labels, Index n_outputs,
        const T *logsumexp, const T *src, const Index *labels, float *val)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < n_outputs)
    {
        __shared__ float block_val;
        if(threadIdx.x == 0)
        {
            block_val = 0;
        }
        float val1 = static_cast<Y>(logsumexp[i]);
        float val2 = static_cast<Y>(src[labels[i] + i*n_labels]);
        atomicAdd(&block_val, val1-val2);
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(val, alpha*block_val);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Scalar alpha, Index n_labels, Index n_outputs,
        const T *logsumexp, const T *src, const int64_t *labels_, float *val)
    noexcept
//! Total sum accumulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 * for every i in [0, n_outputs)
 *      val += alpha * (logsumexp[i]-src[labels[i], i]);
 *
 * @param[in] alpha: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of elements to sum up.
 * @param[in] logsumexp: Array with logsumexp values of size n_outputs.
 * @param[in] src: Matrix of size n_labels times n_outputs stored continuously
 *      in Fortran order
 * @param[in] labels: Array of size n_outputs with correct labels
 * @param[inout] val: Scalar that accumulates the total sum
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    using I = typename CUDAComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(alpha, n_labels,
            n_outputs, logsumexp, src, labels, val);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, const fp32_t* logsumexp, const fp32_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, const fp64_t* logsumexp, const fp64_t* src,
        const int64_t* labels, float *val)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, const bf16_t* logsumexp, const bf16_t* src,
        const int64_t* labels, float *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
