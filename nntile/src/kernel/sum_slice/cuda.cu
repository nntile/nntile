/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_slice/cuda.cu
 * Sums over fibers into a slice of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/sum_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sum_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    Y alpha{alpha_};
    Y beta{beta_};
    constexpr Y zero{0.0};
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src
        const T *src_fiber = src + i1*mk + i0;
        // Init sum over the fiber
        Y sum{zero};
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            sum += Y{src_fiber[i2*m]};
        }
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum);
        __syncthreads();
        // Update output value
        if(i2_start == 0)
        {
            // Output value
            T &result = dst[i1*m+i0];
            sum = block_sum[threadIdx.x+blockDim.x*threadIdx.y];
            if(beta == zero)
            {
                result = T{alpha * sum};
            }
            else
            {
                result = T{beta * Y{result} + alpha * sum};
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided 1-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[0,j] = beta*dst[0,j] + alpha*sum(src[0,:,j])
 *
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous 1-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    // Pointer to a corresponding fiber of the input arrays
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src_fiber = src + src_l + blockIdx.x*k;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y val = static_cast<Y>(src_fiber[c]);
            dst_val += val;
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src_fiber = src + blockIdx.x*k;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y val = static_cast<Y>(src_fiber[c]);
        dst_val += val;
    }
    // Put calculated value into shared memory
    dst_block[threadIdx.x] = alpha * dst_val;
    __syncthreads();
    // Inter-warp reduction
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
        __syncthreads();
    }
    // Reduction within a single warp
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
    }
    // Write output
    if(threadIdx.x == 0)
    {
        if(beta == 0.0)
        {
            dst[blockIdx.x] = static_cast<T>(static_cast<Y>(dst_block[0]));
        }
        else
        {
            dst_val = beta * static_cast<Y>(dst[blockIdx.x]);
            dst[blockIdx.x] = static_cast<T>(dst_val + dst_block[0]);
        }
    }
}

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
static __global__
void cuda_kernel2(Index m, Index n, Index k, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    Index griddim_m = (m+BLOCK_M-1) / BLOCK_M;
    Index block_i = blockIdx.x % griddim_m;
    Index block_j = blockIdx.x / griddim_m;
    Index i = block_i * BLOCK_M;
    Index j = block_j * BLOCK_N;
    __shared__ T src_block[BLOCK_M][BLOCK_K][BLOCK_N];
    __shared__ Y dst_block[BLOCK_M][BLOCK_N];
    // Init dst_block
    for(Index ii = threadIdx.x; ii < BLOCK_M; ii += blockDim.x)
    {
        for(Index jj = 0; jj < BLOCK_N; ++jj)
        {
            dst_block[ii][jj] = 0.0;
        }
    }
    // Proceed by blocks in dimension of k
    for(Index l = 0; l < k; l+= BLOCK_K)
    {
        // Read block of src
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                const T *src_fiber = src + ii + jj*m*k;
                for(Index ll = l; ll < ::min(k, l+BLOCK_K); ++ll)
                {
                    src_block[ii-i][ll-l][jj-j] = src_fiber[ll*m];
                }
            }
        }
        // Now proceed with data in shared memory
        __syncthreads();
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y dst_val = 0.0;
                for(Index ll = l; ll < ::min(k, l+BLOCK_K); ++ll)
                {
                    T val = src_block[ii-i][ll-l][jj-j];
                    dst_val += static_cast<Y>(val);
                }
                dst_block[ii-i][jj-j] += dst_val;
            }
        }
    }
    // Update output
    if(beta == 0.0)
    {
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y val = alpha * dst_block[ii-i][jj-j];
                dst[ii+jj*m] = static_cast<T>(val);
            }
        }
    }
    else
    {
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y val = alpha * dst_block[ii-i][jj-j];
                val += beta * static_cast<Y>(dst[ii+jj*m]);
                dst[ii+jj*m] = static_cast<T>(val);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (cuda_kernel_m1<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src, beta, dst);
    }
    // Case of small k
    else if(k < 1024)
    {
        dim3 threads(256);
        dim3 blocks(((m+255)/256) * n);
        (cuda_kernel2<T, 256, 1, 4>)<<<blocks, threads, 0, stream>>>(m, n, k,
                alpha, src, beta, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src, beta, dst);
    }
}

//! Chunk descriptors for the one-launch chunked sum_slice (see cuda.hh).
template<typename T>
struct SumSliceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    T* const* dst; //!< device-ring view of the dst pointer array
    Index nsrc;
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long ss_chunk_of(long f, Index elems,
    int shift)
{
    return f >> shift;
}

//! Chunk-aware m==1 kernel: the SAME geometry (one block per output column,
//! 256 threads, strided per-thread sum then the shared-memory tree
//! reduction) and the SAME accumulation order as cuda_kernel_m1 above, with
//! each read resolved through the src chunk table and the write through the
//! dst table. Deterministic (no atomics), so bitwise cuda()->m1.
template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_m1_kernel(SumSliceChunkTables<T> t)
{
    Index const elems_src = t.elems_src;
    Index const elems_dst = t.elems_dst;
    int const shift_src = t.shift_src;
    int const shift_dst = t.shift_dst;
    Index const j = blockIdx.x;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    long const base = long(j) * long(t.k);
    long const step = BLOCK_ROW_STEP;
    // The kernel's two original loops (whole BLOCK_ROW blocks, then the
    // tail) together read exactly the offsets tid + step*i for i = 0,1,...
    // while that offset is < k, in increasing i; collapse them into one
    // affine walk and keep the current chunk's base pointer in a register,
    // rebasing only when the walk crosses a chunk boundary.
    long f = base + long(threadIdx.x);
    long const end = base + long(t.k);
    // Run-structured in-chunk walk (piece 4): the span-start divmod runs
    // once per thread; inside a run the element at flat offset f is
    // addressed as q[f] through the virtual contiguous base
    // q = chunk_base - chunk*elems, so the hot loop is branch-free and
    // nvcc pipelines the loads like the classic kernel's loop (a rebase
    // branch inside the walk inhibited unrolling and starved
    // memory-level parallelism). A rebase (one divmod per straddling run
    // end) restores q when the walk crosses into the next chunk. Same
    // values in the same order, so the result is unchanged bitwise.
    if(threadIdx.x < t.k)
    {
        long const c0 = f >> shift_src;
        long const flast = end - 1 - ((end - 1 - f) % step);
        if(flast >> shift_src == c0)
        {
            // Single-run fast path: the plain affine loop through the
            // virtual contiguous base (cuda_kernel_m1's own loop shape,
            // so the loads pipeline like the classic kernel's).
            T const *q = t.src[c0] - c0 * elems_src;
            for(; f < end; f += step)
            {
                dst_val += Y{q[f]};
            }
        }
        else
        {
            long off = f - c0 * elems_src;
            T const *q = t.src[c0] - c0 * elems_src;
            while(f < end)
            {
                long const room = (elems_src - off + step - 1) / step;
                long run_end = f + room * step;
                if(run_end > end)
                {
                    run_end = end;
                }
                for(; f < run_end; f += step)
                {
                    dst_val += Y{q[f]};
                    off += step;
                }
                if(off >= elems_src)
                {
                    long const c = f >> shift_src;
                    off = f - c * elems_src;
                    q = t.src[c] - c * elems_src;
                }
            }
        }
    }
    dst_block[threadIdx.x] = alpha * dst_val;
    __syncthreads();
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
    }
    if(threadIdx.x == 0)
    {
        long const g = long(j);
        long const cd = ss_chunk_of(g, elems_dst, shift_dst);
        T *dp = t.dst[cd] + (g - cd * long(elems_dst));
        if(beta == 0.0)
        {
            dp[0] = T{Y(dst_block[0])};
        }
        else
        {
            Y v = beta * Y{dp[0]};
            dp[0] = T{v + dst_block[0]};
        }
    }
}

//! Chunk-aware general kernel (m>1): the SAME geometry and accumulation
//! order as cuda_kernel above (per-thread strided fiber walk with i2 outer,
//! shared-memory atomicAdd reduction), with reads/writes resolved through
//! the chunk tables. Equal to cuda() up to the atomic ordering floor.
template<typename T>
static __global__
void chunked_general_kernel(SumSliceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    Y alpha{t.alpha};
    Y beta{t.beta};
    constexpr Y zero{0.0};
    if(i0 < t.m and i1 < t.n)
    {
        Y sum{zero};
        long const mk = long(t.m) * long(t.k);
        for(Index i2 = i2_start; i2 < t.k; i2 += i2_step)
        {
            long const f = long(i1)*mk + long(i0) + long(i2)*long(t.m);
            long const ci = ss_chunk_of(f, t.elems_src, t.shift_src);
            sum += Y{t.src[ci][f - ci * long(t.elems_src)]};
        }
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum);
        __syncthreads();
        if(i2_start == 0)
        {
            long const g = long(i1)*long(t.m) + long(i0);
            long const cd = ss_chunk_of(g, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + (g - cd * long(t.elems_dst));
            sum = block_sum[threadIdx.x+blockDim.x*threadIdx.y];
            if(beta == zero)
            {
                dp[0] = T{alpha * sum};
            }
            else
            {
                dp[0] = T{beta * Y{dp[0]} + alpha * sum};
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 or n <= 0 or k <= 0 or nsrc <= 0 or ndst <= 0)
    {
        return;
    }
    SumSliceChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.nsrc = nsrc;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_sum_slice");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_sum_slice");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    // Mirror cuda()'s branch selection and geometry.
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (chunked_m1_kernel<T, 1024, 4>)<<<blocks, threads, 0,
            stream>>>(t);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (chunked_general_kernel<T>)<<<blocks, threads, 0,
            stream>>>(t);
    }
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
        Scalar alpha, const fp32_t* const* src_chunks, Index nsrc, Scalar beta,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp64_t* const* src_chunks, Index nsrc, Scalar beta,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const bf16_t* const* src_chunks, Index nsrc, Scalar beta,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp16_t* const* src_chunks, Index nsrc, Scalar beta,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;






} // namespace nntile::kernel::sum_slice
