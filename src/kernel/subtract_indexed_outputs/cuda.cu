/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subtract_indexed_outputs/cuda.cu
 * subtract_indexed_output operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/subtract_indexed_outputs/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
static __global__
void cuda_kernel(Index n_labels, Index n_outputs, Scalar val_, const Index* labels, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    Y dst_val{0.0};
    Y val{val_};
    if(i < n_outputs)
    {
        dst_val = Y{dst[labels[i] + i*n_labels]};
        dst[labels[i] + i*n_labels] = dst_val - val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, Scalar val,
        const int64_t *labels_, T *dst_)
    noexcept
//! Subtraction of given val from indexed output of dst
/*! Mnemonically, the following operations are performed:
 *      dst[labels[i], i] -= val
 * for every i in [0, n_outputs)
 *
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of matrix elemets to update
 * @param[in] val: Value that is subtracted from the matrix elements
 * @param[in] labels_: Index array of size n_outputs
 * @param[inout] dst_: Matrix of size n_labels by n_outputs continuously stored
 *      in Fortran order
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    using I = typename CUDAComputeType<int64_t>::value;
    auto labels = reinterpret_cast<const I *>(labels_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(n_labels, n_outputs,
            val, labels, dst_);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Scalar val, const int64_t *labels, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Scalar val, const int64_t *labels, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index n_labels, Index n_outputs,
        Scalar val, const int64_t *labels, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::subtract_indexed_outputs
