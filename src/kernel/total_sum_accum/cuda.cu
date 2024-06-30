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
 * @version 1.0.0
 * */

#include "nntile/kernel/total_sum_accum/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::total_sum_accum
{

template<typename T>
static __global__
void cuda_kernel(T alpha, Index n_labels, Index n_outputs, const T* logsumexp,
        const T* src, const Index* labels, T *val)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < n_outputs)
    {
         atomicAdd(val, alpha*(logsumexp[i]-src[labels[i]+i*n_labels]));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Scalar alpha, Index n_labels, Index n_outputs,
        const T *logsumexp_, const T *src_, const int64_t *labels_, T *val_)
    noexcept
//! Total sum accumulating from logsumexp and corrected by elements from src
/*! Mnemonically, the following operations are performed:
 * for every i in [0, n_outputs)
 *      val += alpha * (logsumexp[i]-src[labels[i], i]);
 *
 * @param[in] alpha: Scalar multiplier
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of elements to sum up.
 * @param[in] logsumexp_: Array with logsumexp values of size n_outputs.
 * @param[in] src_: Matrix of size n_labels times n_outputs stored continuously
 *      in Fortran order
 * @param[in] labels_: Array of size n_outputs with correct labels
 * @param[inout] val_: Scalar that accumulates the total sum
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    using Y = typename CUDAComputeType<T>::value;
    using I = typename CUDAComputeType<int64_t>::value;
    auto logsumexp = reinterpret_cast<const Y *>(logsumexp_);
    auto src = reinterpret_cast<const Y *>(src_);
    auto labels = reinterpret_cast<const I *>(labels_);
    auto val = reinterpret_cast<Y *>(val_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(Y{alpha}, n_labels,
            n_outputs, logsumexp, src, labels, val);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, const fp32_t* logsumexp, const fp32_t* src,
        const int64_t* labels, fp32_t *val)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Scalar alpha, Index n_labels,
        Index n_outputs, const fp64_t* logsumexp, const fp64_t* src,
        const int64_t* labels, fp64_t *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
