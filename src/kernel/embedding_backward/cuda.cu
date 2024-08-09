/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding_backward/cuda.cu
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/embedding_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::embedding_backward
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *embed_, T *vocab)
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
 * @param[out] embed_: Tensor of gradients of embeddings
 * @param[inout] vocab: Gradient of vocabulary. It is a contiguous matrix of
 *      shape (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0 = blockIdx.y, i1 = blockIdx.z;
    using Z = typename CUDAComputeType<T>::value;
    if(i2 < k_size)
    {
        // Output slice of vocabulary
        Z *vocab_slice = reinterpret_cast<Z*>(vocab + k_size*index[i1*m+i0]);
        const Z *embed = reinterpret_cast<const Z *>(embed_);
        // Update value of embedding
        atomicAdd(&vocab_slice[i2], embed[(i1*k+k_start+i2)*m + i0]);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, const int64_t *index_, const T *embed, T *vocab)
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
 * @param[in] index_: Tokens (indices of embeddings)
 * @param[out] embed: Tensor of gradients of embeddings
 * @param[inout] vocab: Gradient of vocabulary. It is a contiguous matrix of
 *      shape (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(256, 1, 1);
    dim3 blocks((k_size+255)/256, m, n);
    using I = typename CUDAComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start, k_size,
            index, embed, vocab);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const fp32_t *embed,
        fp32_t *vocab)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const fp64_t *embed,
        fp64_t *vocab)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const bf16_t *embed,
        bf16_t *vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
