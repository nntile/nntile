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
        Scalar alpha, const Index *index, const T *embed_, T *vocab)
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0 = blockIdx.y, i1 = blockIdx.z;
    using Z = typename CUDAComputeType<T>::value;
    using Y = typename T::repr_t;
    if(i2 < k_size)
    {
        Z *vocab_slice = reinterpret_cast<Z*>(vocab + k_size*index[i1*m+i0]);
        const Z *embed = reinterpret_cast<const Z *>(embed_);
        const Y contrib = static_cast<Y>(alpha) *
            static_cast<Y>(embed[(i1*k+k_start+i2)*m + i0]);
        atomicAdd(&vocab_slice[i2], static_cast<Z>(contrib));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index vocab_nelems, Scalar alpha, Scalar beta,
        const int64_t *index_, const T *embed, T *vocab)
    noexcept
{
    if(beta == Scalar{0.0})
    {
        cudaMemsetAsync(vocab, 0, static_cast<size_t>(vocab_nelems) * sizeof(T),
                stream);
    }
    dim3 threads(256, 1, 1);
    dim3 blocks((k_size+255)/256, m, n);
    using I = typename CUDAComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, k_start, k_size,
            alpha, index, embed, vocab);
}

template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index, const fp32_t *embed, fp32_t *vocab)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index, const fp64_t *embed, fp64_t *vocab)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index, const bf16_t *embed, bf16_t *vocab)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index k_start, Index k_size, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index, const fp16_t *embed, fp16_t *vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
