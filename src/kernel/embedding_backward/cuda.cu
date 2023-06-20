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
    Index i0_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_start = threadIdx.z + blockIdx.z*blockDim.z,
          i0_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y,
          i2_step = blockDim.z * gridDim.z;
    // Cycle over column of embed buffer
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Cycle over row of embed buffer
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            // Output slice of vocabulary
            T *vocab_slice = vocab + k_size*index[i2*m+i1];
            // Input slice of embedding
            const T *embed_slice = embed + (i2*k+k_start)*m + i1;
            // Cycle over slice of output vocab
            for(Index i0 = i0_start; i0 < k_size; i0 += i0_step)
            {
                vocab_slice[i0] += embed_slice[i0*m];
            }
        }
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
    dim3 blocks(8, 8, 8), threads(4, 2, 4);
    if(m == 1)
    {
        blocks = dim3(16, 1, 16);
        threads = dim3(8, 1, 4);
    }
    else if(n == 1)
    {
        blocks = dim3(16, 16);
        threads = dim3(8, 4);
    }
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

