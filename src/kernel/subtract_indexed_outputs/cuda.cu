/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/pow/cuda.cu
 * subtract_indexed_output operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/subtract_indexed_outputs/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_outputs
{

template<typename T>
static __global__
void cuda_kernel(Index n_labels, Index n_outputs, T val, const Index* labels, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < n_outputs)
    {
         dst[labels[i] + i*n_labels] -= val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, T val, const Index* labels, T *dst)
    noexcept
//! Subtraction of given val from indexed output of dst
/*! Mnemonically, the following operations are performed:
 *      dst[labels[i], i] -= val
 * for every i in [0, n_outputs)
 *
 * @param[in] n_labels: Number of possible labels
 * @param[in] n_outputs: Number of matrix elemets to update
 * @param[in] val: Value that is subtracted from the matrix elements
 * @param[in] labels: Index array of size n_outputs
 * @param[inout] dst: Matrix of size n_labels by n_outputs continuously stored
 *      in Fortran order
 * */
{
    dim3 blocks((n_outputs+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(n_labels, n_outputs, val, labels, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index n_labels, Index n_outputs, fp32_t val, const Index* labels, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index n_labels, Index n_outputs, fp64_t val, const Index* labels, fp64_t *dst)
    noexcept;

} // namespace subtract_indexed_outputs
} // namespace kernel
} // namespace nntile

