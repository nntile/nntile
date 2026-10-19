/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/multiply_slice/cuda.cu
 * Per-element multiplication of a tensor by a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/multiply_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::multiply_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_, const T *src,
        T *dst)
//! Per-element product of a tensor and a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] mk: Product of m and k
 * @param[in] alpha_: Scalar factor
 * @param[in] src: Input contiguous m-by-n array
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i0 < m and i1 < n and i2 < k)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to multiply by the output fiber
        const Y src_val = alpha * Y{src[i1*m+i0]};
        // Update output value
        dst_fiber[i2*m] = T{Y{dst_fiber[i2*m]} * src_val};
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src, T *dst)
//! Per-element product of a tensor and a broadcasted slice on CUDA
/*! This is a global function that does the following operations:
 *      dst[0,l,j] = alpha * dst[0,l,j] * src[0,j]
 *
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src: Input contiguous 1-by-n array
 * @param[inout] dst: Input and output contiguous 1-by-k-by-n array
 * */
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    __shared__ Y src_block[BLOCK_COL];
    __shared__ T dst_block[BLOCK_ROW][BLOCK_COL];
    Index src_j = dst_block_j*BLOCK_COL + threadIdx.x;
    if(src_j < n and threadIdx.x < BLOCK_COL)
    {
        src_block[threadIdx.x] = alpha * static_cast<Y>(src[src_j]);
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = dst_fiber[c*k];
            }
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src_block[c] *
                        static_cast<Y>(dst_block[dst_l][c]));
            }
        }
        else
        {
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = dst_fiber[c*k];
            }
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src_block[c] *
                        static_cast<Y>(dst_block[dst_l][c]));
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, T *dst)
    noexcept
//! Per-element product of a tensor and a broadcasted slice on CUDA
/*! This is a host function that does the following operations:
 *      dst[i,l,j] = alpha * dst[i,l,j] * src[i,j]
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src: Input contiguous m-by-n array
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Custom version of code for special m==1 case
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src, dst);
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src, dst);
    }
}

//! Chunk descriptors for the one-launch chunked multiply_slice (see cuda.hh):
//! ONLY the two base-pointer tables travel in the struct; the geometry
//! (m, n, k, alpha) and the chunk-size scalars are separate kernel
//! parameters, kept in registers so the hot loop never reloads them from
//! local memory (the array members force the struct itself into local
//! memory).
template<typename T>
struct ChunkTables
{
    const T *const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T *const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Resolve src's flat offset f to its home pointer via the chunk table.
template<typename T>
static __device__ __forceinline__ T const *chunked_src_ptr(const T *const *src,
    Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return src[c] + (f - c * long(elems));
}

//! Resolve dst's flat offset f to its home pointer via the chunk table.
template<typename T>
static __device__ __forceinline__ T *chunked_dst_ptr(T *const *dst,
    Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return dst[c] + (f - c * long(elems));
}

//! Chunk-aware generic multiply_slice kernel: exactly the classic
//! cuda_kernel geometry ((i0, i1, i2) thread assignment), but each element
//! resolves src's flat slice offset i1*m + i0 and dst's flat offset
//! i1*m*k + i0 + i2*m through the chunk tables. The per-element arithmetic
//! is exactly the classic kernel's, so the result is bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index m, Index n, Index k, Scalar alpha_,
    Index elems_src, int shift_src, Index elems_dst, int shift_dst,
    ChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i0 < m and i1 < n and i2 < k)
    {
        long const fs = long(i1)*m + i0;
        long const fd = long(i1)*(m*k) + i0 + long(i2)*m;
        const Y src_val = alpha *
            Y{*chunked_src_ptr(t.src, elems_src, shift_src, fs)};
        T *dp = chunked_dst_ptr(t.dst, elems_dst, shift_dst, fd);
        *dp = T{Y{*dp} * src_val};
    }
}

//! Chunk-aware m==1 multiply_slice kernel: exactly the classic
//! cuda_kernel_m1 geometry, but each src/dst element is resolved through its
//! chunk table. The per-element arithmetic is exactly the classic kernel's,
//! so the result is bitwise cuda()'s m==1 branch.
template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void chunked_cuda_kernel_m1(Index n, Index k, Scalar alpha_,
    Index elems_src, int shift_src, Index elems_dst, int shift_dst,
    ChunkTables<T> t)
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    __shared__ Y src_block[BLOCK_COL];
    __shared__ T dst_block[BLOCK_ROW][BLOCK_COL];
    Index src_j = dst_block_j*BLOCK_COL + threadIdx.x;
    if(src_j < n and threadIdx.x < BLOCK_COL)
    {
        // m == 1: src's flat offset of element j is j * 1 = j.
        src_block[threadIdx.x] = alpha * static_cast<Y>(
            *chunked_src_ptr(t.src, elems_src, shift_src, long(src_j)));
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        // Classic two-phase staging (load dst into shared, then write);
        // dst_block[dst_l] is thread-private (dst_l == threadIdx.x for
        // the m==1 launch geometry), so no barrier is needed between the
        // phases -- exactly like the classic kernel. Piece 4: this
        // thread's columns live at the flat offsets fe0 + c*k, a
        // per-thread walk; resolve the dst home pointer ONCE through the
        // virtual contiguous base q = chunk_base - chunk*elems (the
        // element at flat f is q[f]) and rebase only when the walk
        // crosses a chunk boundary -- the per-element table resolution
        // (a 64-bit divmod whenever the chunk size is not a power of
        // two, which is every 1-chunk operand) sat in the middle of both
        // column loops. Same elements in the same order, so the result
        // is unchanged bitwise.
        Index const ncols = (dst_block_j+1)*BLOCK_COL <= n
            ? Index(BLOCK_COL) : n - global_dst_j;
        long const fe0 = long(global_dst_l) + long(global_dst_j)*long(k);
        long const step_col = long(BLOCK_COL_STEP) * long(k);
        // Single-run fast path: this thread's whole column range inside
        // one dst chunk (every 1-chunk operand, and interior spans) ->
        // the classic two-phase loops verbatim through the virtual
        // contiguous base (the element at flat f is q[f]).
        long const c_d0 = fe0 >> shift_dst;
        if((fe0 + long(ncols - 1) * long(k)) >> shift_dst == c_d0)
        {
            T *qd = t.dst[c_d0] - c_d0 * elems_dst;
            for(int c = 0; c < ncols; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = qd[fe0 + long(c) * long(k)];
            }
            for(int c = 0; c < ncols; c += BLOCK_COL_STEP)
            {
                qd[fe0 + long(c) * long(k)] = static_cast<T>(src_block[c] *
                    static_cast<Y>(dst_block[dst_l][c]));
            }
        }
        else
        for(int phase = 0; phase < 2; ++phase)
        {
            long c_d = fe0 >> shift_dst;
            long off_d = fe0 - c_d * elems_dst;
            T *qd = t.dst[c_d] - c_d * elems_dst;
            Index c = 0;
            long fe = fe0;
            while(c < ncols)
            {
                // Columns before the next chunk boundary.
                long const room = (elems_dst - off_d + step_col - 1)
                    / step_col;
                long const run = room * BLOCK_COL_STEP;
                Index const run_cols = run < long(ncols - c)
                    ? Index(run) : ncols - c;
                for(Index i = 0; i < run_cols; i += BLOCK_COL_STEP)
                {
                    if(phase == 0)
                    {
                        dst_block[dst_l][c] = qd[fe];
                    }
                    else
                    {
                        qd[fe] = static_cast<T>(src_block[c] *
                            static_cast<Y>(dst_block[dst_l][c]));
                    }
                    fe += step_col;
                    off_d += step_col;
                    c += BLOCK_COL_STEP;
                }
                if(off_d >= elems_dst)
                {
                    c_d = fe >> shift_dst;
                    off_d = fe - c_d * elems_dst;
                    qd = t.dst[c_d] - c_d * elems_dst;
                }
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
    const T *const *src_chunks, Index nsrc, Index elems_src,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    int const shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_multiply_slice");
    int const shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_multiply_slice");
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (chunked_cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0,
            stream>>>(n, k, alpha, elems_src, shift_src, elems_dst, shift_dst,
            t);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
            m, n, k, alpha, elems_src, shift_src, elems_dst, shift_dst, t);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *const *src_chunks, Index nsrc, Index elems_src,
        fp32_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *const *src_chunks, Index nsrc, Index elems_src,
        fp64_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *const *src_chunks, Index nsrc, Index elems_src,
        bf16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *const *src_chunks, Index nsrc, Index elems_src,
        fp16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;






} // namespace nntile::kernel::multiply_slice
