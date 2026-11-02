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
//! Piece 4, residual parity: the classic kernel pays one block-wide
//! __syncthreads to broadcast the staged alpha*src row (BLOCK_COL values)
//! through shared memory, and stages dst through shared behind the same
//! barrier. The chunked kernel cannot stage src that cheaply -- its staging
//! load runs through the chunk table (a dependent table->element load chain
//! that stretched every block's barrier wait), and dst staging via the
//! per-thread chunk walk also had to resolve the table. Since dst rows are
//! thread-private in this geometry (dst_l == threadIdx.x) and every thread
//! of a block reads the SAME BLOCK_COL src values, the shared staging buys
//! nothing that L1 does not: resolving each operand's home chunk ONCE per
//! thread through the virtual contiguous base (the element at flat f is
//! q[f]) and reading src per thread directly (L1-resident 32 bytes) removes
//! the barrier, the shared arrays and the staging phases completely. The
//! full-row fast path is a straight line of BLOCK_COL independent
//! load/multiply/store triples with the compile-time trip count, unrolled
//! and register-resident; straddling spans and ragged tails fall back to a
//! per-column walk with per-column chunk resolution. Same per-element
//! expression in the same order, so the result is unchanged bitwise.
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
    if(global_dst_l >= k)
    {
        return;
    }
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
    // Full row of columns vs ragged tail of the last block column.
    bool const full = ((dst_block_j+1)*BLOCK_COL <= n);
    Index const ncols = full ? Index(BLOCK_COL) : n - global_dst_j;
    // m == 1: this thread's columns live at dst's flat offsets
    // fe0 + c*k and read src's flat offsets fs0 + c (c in [0, ncols)).
    // Resolve each operand's home pointer ONCE through its virtual
    // contiguous base (the element at flat f is q[f]); rebase only when
    // the walk crosses a chunk boundary -- the per-element table
    // resolution sat in the middle of the column loops.
    long const fe0 = long(global_dst_l) + long(global_dst_j)*long(k);
    long const c_d0 = fe0 >> shift_dst;
    T *qd = t.dst[c_d0] - c_d0 * elems_dst;
    long const fs0 = long(global_dst_j);
    long const c_s0 = fs0 >> shift_src;
    T const *qs = t.src[c_s0] - c_s0 * elems_src;
    // Single-run fast path: this thread's whole column range stays inside
    // one dst chunk AND one src chunk (every 1-chunk operand, and interior
    // spans) -> barrier-free straight line; the full-row branch carries the
    // compile-time trip count BLOCK_COL exactly like the classic kernel's
    // unrolled fast branch. Per-element expression and evaluation order are
    // the classic kernel's: dst = T{dst * (alpha * src)}.
    if((fe0 + long(ncols - 1) * long(k)) >> shift_dst == c_d0
        and (fs0 + long(ncols - 1)) >> shift_src == c_s0)
    {
        if(full)
        {
#pragma unroll
            for(int c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                qd[fe0 + c*k] = static_cast<T>(static_cast<Y>(qd[fe0 + c*k])
                    * (alpha * static_cast<Y>(qs[fs0 + c])));
            }
        }
        else
        {
            for(Index c = 0; c < ncols; c += BLOCK_COL_STEP)
            {
                qd[fe0 + c*k] = static_cast<T>(static_cast<Y>(qd[fe0 + c*k])
                    * (alpha * static_cast<Y>(qs[fs0 + c])));
            }
        }
    }
    else
    {
        // Column walk with per-column chunk resolution: rows whose dst or
        // src span straddles a chunk boundary, and any ragged tail.
        for(Index c = 0; c < ncols; c += BLOCK_COL_STEP)
        {
            long const fs = fs0 + c;
            long const c_s = fs >> shift_src;
            T const *sp = t.src[c_s] + (fs - c_s * elems_src);
            long const fe = fe0 + c*k;
            long const c_d = fe >> shift_dst;
            T *dp = t.dst[c_d] + (fe - c_d * elems_dst);
            *dp = static_cast<T>(static_cast<Y>(*dp)
                * (alpha * static_cast<Y>(*sp)));
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
