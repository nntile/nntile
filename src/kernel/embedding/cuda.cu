/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding/cuda.cu
 * Embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/embedding/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::embedding
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index k_start, Index k_size,
        const Index *index, const T *vocab, T *embed)
//! Fill embedding from vocabulary
/*! Fill provided m-by-k-by-n output tensor embed:
 *      embed[i, k_start:k_start+k_size, j] = vocab[:, index[i, j]]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index: Tokens (indices of embeddings)
 * @param[in] vocab: Vocabulary of embeddings. It is a contiguous matrix of shape
 *      (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * @param[inout] embed: Output tensor to be filled with embeddings
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    if(i2 < k_size and i1 < n and i0 < m)
    {
            // Input slice of vocabulary
            const T *vocab_slice = vocab + k_size*index[i1*m+i0];
            // Output value to be updated
            embed[(i1*k+k_start+i2)*m + i0] = vocab_slice[i2];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, const int64_t *index_, const T *vocab, T *embed)
    noexcept
//! Fill embedding from vocabulary
/*! Fill provided m-by-k-by-n output tensor embed:
 *      embed[i, k_start:k_start+k_size, j] = vocab[:, index[i, j]]
 *
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index_: Tokens (indices of embeddings)
 * @param[in] vocab: Vocabulary of embeddings. It is a contiguous matrix of shape
 *      (k_size, vocab_size) but vocab_size is not passed as a parameter.
 * @param[inout] embed: Output tensor to be filled with embeddings
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k_size), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k_size+threads.z-1)/threads.z);
    using I = typename CUDAComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start, k_size,
            index, vocab, embed);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const fp32_t *vocab,
        fp32_t *embed)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const bf16_t *vocab,
        bf16_t *embed)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, const int64_t *index, const fp64_t *vocab,
        fp64_t *embed)
    noexcept;

} // namespace nntile::kernel::embedding
