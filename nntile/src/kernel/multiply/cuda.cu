/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/multiply/cuda.cu
 * Per-element product of two buffers on CUDA
 *
 * @version 2.0.0
 * */

#include "nntile/kernel/multiply/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::multiply
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T *src1, const T *src2, T *dst)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    __shared__ T src1_block[BLOCK];
    __shared__ T src2_block[BLOCK];
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src1[i+j];
            src2_block[threadIdx.x+j] = src2[i+j];
        }
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    alpha * static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
    else
    {
        // Per-thread clamp: threads of the last (partial) block whose
        // index reached nelems must not read or write. The old
        // block-level bound let every thread of the partial block run up
        // to the block's end, reading and writing up to BLOCK-1 elements
        // past nelems - tolerated by over-allocated test buffers, but out
        // of bounds for an exact-size chunk of a chunked tile.
        int j_max = nelems - i;
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src1[i+j];
            src2_block[threadIdx.x+j] = src2[i+j];
        }
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    alpha * static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src1, const T *src2,
        T *dst)
    noexcept
//! Per-element product of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] alpha: Scalar multiplier
 * @param[in] src1: Input buffer
 * @param[in] src2: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src1,
            src2, dst);
}


template<typename T, int BLOCK, int LOOP>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha, ChunkConst<T> src1,
        ChunkConst<T> src2, ChunkMut<T> dst)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    __shared__ T src1_block[BLOCK];
    __shared__ T src2_block[BLOCK];
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src1.at(i+j)[0];
            src2_block[threadIdx.x+j] = src2.at(i+j)[0];
        }
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            dst.at(i+j)[0] = static_cast<T>(
                    alpha * static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
    else
    {
        int j_max = nelems - i;
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src1.at(i+j)[0];
            src2_block[threadIdx.x+j] = src2.at(i+j)[0];
        }
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            dst.at(i+j)[0] = static_cast<T>(
                    alpha * static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1,
        const T* const* src2_chunks, Index nsrc2,
        T* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || nsrc1 <= 0 || nsrc2 <= 0 || ndst <= 0)
    {
        return;
    }
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (chunked_cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems,
        alpha,
        chunk_const<T>(src1_chunks, nsrc1, elems_src1, cls, "nntile_multiply"),
        chunk_const<T>(src2_chunks, nsrc2, elems_src2, cls, "nntile_multiply"),
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_multiply"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t *src1,
        const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_fast_tf32_t *src1, const fp32_fast_tf32_t *src2,
        fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_fast_fp16_t *src1, const fp32_fast_fp16_t *src2,
        fp32_fast_fp16_t *dst)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_fast_bf16_t *src1, const fp32_fast_bf16_t *src2,
        fp32_fast_bf16_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t *src1,
        const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t *src1,
        const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t *src1,
        const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src1_chunks, Index nsrc1,
        const fp32_t* const* src2_chunks, Index nsrc2,
        fp32_t* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src1_chunks, Index nsrc1,
        const fp64_t* const* src2_chunks, Index nsrc2,
        fp64_t* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src1_chunks, Index nsrc1,
        const bf16_t* const* src2_chunks, Index nsrc2,
        bf16_t* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src1_chunks, Index nsrc1,
        const fp16_t* const* src2_chunks, Index nsrc2,
        fp16_t* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::multiply
