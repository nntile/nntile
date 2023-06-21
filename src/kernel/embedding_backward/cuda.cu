/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding_backward/cuda.cu
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#include "nntile/kernel/embedding_backward/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace embedding_backward
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *embed, T *vocab)
//! Accumulate gradients of embeddings into vocabulary
/*! Does the following operation:
 *      vocab[:, index[i, j]] += embed[i, k_start:k_start+k_size, j]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[out] embed: Tensor of gradients of embeddings
 * @param[inout] vocab: Gradient of vocabulary. It is a contiguous matrix of
 *      shape (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i2 < k_size and i1 < n and i0 < m)
    {
        // Output slice of vocabulary
        T *vocab_slice = vocab + k_size*index[i1*m+i0];
        // Update value of embedding
        vocab_slice[i2] += embed[(i1*k+k_start+i2)*m + i0];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, const Index *index, const T *embed, T *vocab)
    noexcept
//! Accumulate gradients of embeddings into vocabulary
/*! Does the following operation:
 *      vocab[:, index[i, j]] += embed[i, k_start:k_start+k_size, j]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[out] embed: Tensor of gradients of embeddings
 * @param[inout] vocab: Gradient of vocabulary. It is a contiguous matrix of
 *      shape (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k_size), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k_size+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start, k_size,
            index, embed, vocab);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const Index *index, const fp32_t *embed,
        fp32_t *vocab)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const Index *index, const fp64_t *embed,
        fp64_t *vocab)
    noexcept;

} // namespace embedding_backward
} // namespace kernel
} // namespace nntile

