/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice_inplace/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_slice_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_slice_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
//! Generic implementation of the add_slice_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_slice_inplace::cuda
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to add to the output fiber
        const Y src_val = Y{alpha} * Y{src[i1*m+i0]};
        // Read value from the output
        T &dst_val = dst_fiber[i2*m];
        // And update it
        dst_val = T{beta * Y{dst_val} + src_val};
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Generic implementation of the add_slice_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_slice_inplace::cuda
 * */
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
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
                dst_fiber[c*k] = static_cast<T>(
                        beta*static_cast<Y>(dst_block[dst_l][c]) +
                        src_block[c]);
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
                dst_fiber[c*k] = static_cast<T>(
                        beta*static_cast<Y>(dst_block[dst_l][c]) +
                        src_block[c]);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src_, Scalar beta, T *dst_)
    noexcept
//! Add a broadcasted slice to a tensor with optional scaling inplace on CUDA
/*! Performs the following operation:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[i,j] is NaN, then dst[i,l,j] will be NaN.
 * If beta is zero and dst[i,l,j] is NaN, then dst[i,l,j] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace instead, and in a case of beta being zero,
 * use nntile::kernel::scale_slice instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale_slice
 * @see nntile::kernel::clear
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta_: Scaling factor for dst
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
                alpha, src_, beta, dst_);
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src_, beta, dst_);
    }
}

//! Chunk descriptors for the one-launch chunked add_slice_inplace (cuda.hh).
template<typename T>
struct AddSliceInplaceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T* const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< log2(elems_src) on a power of two, else -1
    int shift_dst; //!< log2(elems_dst) on a power of two, else -1
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index adsli_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adsli_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adsli_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long adsli_off(long f, Index elems,
    int shift, Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware add_slice_inplace generic (m > 1) path: the SAME thread
//! geometry and per-element expression as cuda_kernel, every operand
//! address resolved through its chunk table (src flat i1*m+i0, dst flat
//! i1*m*k+i2*m+i0).
template<typename T>
static __global__
void chunked_cuda_kernel(AddSliceInplaceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y beta = t.beta;
    const Y alpha = t.alpha;
    if(i2 < t.k and i1 < t.n and i0 < t.m)
    {
        long const f1 = long(i1)*long(t.m) + long(i0);
        Index const cs = adsli_chunk_of(f1, t.elems_src, t.shift_src);
        const Y sv = Y{t.src[cs][adsli_off(f1, t.elems_src, t.shift_src, cs)]};
        long const f2 = long(i1)*long(t.m)*long(t.k) + long(i2)*long(t.m)
            + long(i0);
        Index const cd = adsli_chunk_of(f2, t.elems_dst, t.shift_dst);
        T *dp = t.dst[cd] + adsli_off(f2, t.elems_dst, t.shift_dst, cd);
        // Same association as the generic cuda_kernel: fl(beta*dst), then
        // fma(alpha, src, .).
        Y pr = beta * Y{dp[0]};
        dp[0] = T{adsli_fma(alpha, sv, pr)};
    }
}

//! Chunk-aware add_slice_inplace m == 1 path: the SAME BLOCK_ROW x BLOCK_COL
//! tiling and per-element expression as cuda_kernel_m1 (src and dst flat
//! indices are j and j*k+l respectively when m == 1), every address resolved
//! through its chunk table. The m == 1 tiling is essential: the generic 3D
//! grid has threadDim.x == m == 1, so its (i2) stride is non-coalesced.
template<typename T, int BLOCK_ROW, int BLOCK_COL>
static __global__
void chunked_cuda_kernel_m1(AddSliceInplaceChunkTables<T> t)
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (t.k + BLOCK_ROW - 1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y beta = t.beta;
    const Y alpha = t.alpha;
    __shared__ Y src_block[BLOCK_COL];
    Index src_j = dst_block_j*BLOCK_COL + threadIdx.x;
    if(threadIdx.x < BLOCK_COL)
    {
        if(src_j < t.n)
        {
            long const f1 = long(src_j); // m == 1
            Index const cs = adsli_chunk_of(f1, t.elems_src, t.shift_src);
            src_block[threadIdx.x] = alpha * Y{
                t.src[cs][adsli_off(f1, t.elems_src, t.shift_src, cs)]};
        }
        else
        {
            src_block[threadIdx.x] = Y{0.0};
        }
    }
    __syncthreads();
    if(global_dst_l < t.k)
    {
        Index const c_end = (dst_block_j + 1)*BLOCK_COL <= t.n
            ? BLOCK_COL : t.n - global_dst_j;
        for(Index c = 0; c < c_end; ++c)
        {
            long const f2 = long(global_dst_j + c)*long(t.k)
                + long(global_dst_l);
            Index const cd = adsli_chunk_of(f2, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + adsli_off(f2, t.elems_dst, t.shift_dst, cd);
            // Same association as cuda_kernel_m1: fl(alpha*src) (in
            // shared src_block), then fma(beta, dst, .).
            dp[0] = T{adsli_fma(beta, Y{dp[0]}, src_block[c])};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    AddSliceInplaceChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_add_slice_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add_slice_inplace");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    if(m == 1)
    {
        // Mirror cuda()'s m == 1 dispatch: BLOCK_ROW x BLOCK_COL tiling.
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (chunked_cuda_kernel_m1<T, 256, 8>)<<<blocks, threads, 0,
            stream>>>(t);
        return;
    }
    dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
            std::min(int(k), 16));
    dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
            (k+threads.z-1)/threads.z);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_t* const* src_chunks, Index nsrc,
        Scalar beta, fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp64_t* const* src_chunks, Index nsrc,
        Scalar beta, fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const bf16_t* const* src_chunks, Index nsrc,
        Scalar beta, bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp16_t* const* src_chunks, Index nsrc,
        Scalar beta, fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;






} // namespace nntile::kernel::add_slice_inplace
