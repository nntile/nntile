/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/multiply_inplace/cuda.cu
 * Per-element product of two buffers on CUDA
 *
 * @version 1.2.0
 * */

#include <cstdio>
#include "nntile/kernel/multiply_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::multiply_inplace
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T *src, T *dst)
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
            src1_block[threadIdx.x+j] = src[i+j];
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(alpha *
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
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
            src1_block[threadIdx.x+j] = src[i+j];
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(alpha *
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src, T *dst)
    noexcept
//! Per-element product of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src,
            dst);
}

//! Chunk descriptors for the one-launch chunked multiply_inplace (see
//! cuda.hh). `src`/`dst` are device-ring views of host pointer arrays
//! built by the driver through the shared ChunkPtrRing (no chunk-count
//! cap); each thread reads its chunk pointer straight from the ring
//! (piece 9 / rule 10: no staging), so the whole flat range is one
//! launch.
template<typename T>
struct ChunkTables
{
    const T *const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T *const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

template<typename T>
static __device__ __forceinline__ T chunked_load_src(const T *const *sh,
    Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return sh[c][f - c * long(elems)];
}

template<typename T>
static __device__ __forceinline__ T chunked_load_dst(T *const *sh,
    Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return sh[c][f - c * long(elems)];
}

template<typename T>
static __device__ __forceinline__ void chunked_store_dst(T *const *sh,
    Index elems, int shift, long f, T v) noexcept
{
    long const c = chunk_index(f, elems, shift);
    sh[c][f - c * long(elems)] = v;
}

//! Chunk-aware multiply_inplace kernel: exactly the classic cuda_kernel
//! geometry (BLOCK/LOOP shared-memory blocks, per-thread clamp on the last
//! partial block), but each operand is addressed through its chunk table.
//! Element f = i + j resolves its src chunk by f / elems_src and its dst
//! chunk by f / elems_dst (shifts when those counts are powers of two).
//! The per-element arithmetic is exactly the classic kernel's, so the
//! result is bitwise the whole-buffer cuda() call.
template<typename T, int BLOCK, int LOOP>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha, ChunkTables<T> t)
{
    long i = long(threadIdx.x) + long(blockIdx.x)*BLOCK;
    using Y = typename T::repr_t;
    __shared__ T src1_block[BLOCK];
    __shared__ T src2_block[BLOCK];
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((long(blockIdx.x)+1)*BLOCK <= long(nelems))
    {
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = chunked_load_src(t.src,
                t.elems_src, t.shift_src, i+j);
            src2_block[threadIdx.x+j] = chunked_load_dst(t.dst,
                t.elems_dst, t.shift_dst, i+j);
        }
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            chunked_store_dst(t.dst, t.elems_dst, t.shift_dst, i+j,
                static_cast<T>(alpha *
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j])));
        }
    }
    else
    {
        long j_max = long(nelems) - i;
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = chunked_load_src(t.src,
                t.elems_src, t.shift_src, i+j);
            src2_block[threadIdx.x+j] = chunked_load_dst(t.dst,
                t.elems_dst, t.shift_dst, i+j);
        }
        for(int j = 0; j < j_max; j += BLOCK_STEP)
        {
            chunked_store_dst(t.dst, t.elems_dst, t.shift_dst, i+j,
                static_cast<T>(alpha *
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j])));
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
    const T *const *src_chunks, Index nsrc, Index elems_src,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_multiply_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_multiply_inplace");
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (chunked_cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0,
        stream>>>(nelems, alpha, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t *src,
        fp32_t *dst)
    noexcept;
template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t *src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t *src,
        bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t *src,
        fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *const *src_chunks, Index nsrc, Index elems_src,
        fp32_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *const *src_chunks, Index nsrc, Index elems_src,
        fp64_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *const *src_chunks, Index nsrc, Index elems_src,
        bf16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *const *src_chunks, Index nsrc, Index elems_src,
        fp16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;





} // namespace nntile::kernel::multiply_inplace
