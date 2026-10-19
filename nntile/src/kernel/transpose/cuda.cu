/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/transpose/cuda.cu
 * Transpose operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/transpose/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"
#include "nntile/kernel/scale/cuda.hh"

namespace nntile::kernel::transpose
{

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel(Index m, Index n, Scalar alpha_, const T *src, T *dst)
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha_: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    Index src_i = threadIdx.x % BLOCK_ROW;
    Index src_j = threadIdx.x / BLOCK_ROW;
    Index src_griddim_row = (m+BLOCK_ROW-1) / BLOCK_ROW;
    Index src_block_i = blockIdx.x % src_griddim_row;
    Index src_block_j = blockIdx.x / src_griddim_row;
    Index global_src_i = src_i + src_block_i*BLOCK_ROW;
    Index global_src_j = src_j + src_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};

    if((src_block_i+1)*BLOCK_ROW <= m and (src_block_j+1)*BLOCK_COL <= n)
    {
        __shared__ T block[BLOCK_ROW][BLOCK_COL+1];
        const T *src_slice = src + global_src_i + global_src_j*m;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        for(int k = 0; k < BLOCK_COL; k += BLOCK_COL_STEP)
        {
            block[src_i][src_j+k] = T{alpha * Y{src_slice[k*m]}};
        }
        Index dst_i = threadIdx.x % BLOCK_COL;
        Index dst_j = threadIdx.x / BLOCK_COL;
        Index dst_block_i = src_block_j;
        Index dst_block_j = src_block_i;
        Index global_dst_i = dst_i + dst_block_i*BLOCK_COL;
        Index global_dst_j = dst_j + dst_block_j*BLOCK_ROW;
        T *dst_slice = dst + global_dst_i + global_dst_j*n;
        __syncthreads();
        constexpr int BLOCK_ROW_STEP = BLOCK_ROW / BLOCK_LOOP;
        for(int k = 0; k < BLOCK_ROW; k += BLOCK_ROW_STEP)
        {
            dst_slice[k*n] = block[dst_j+k][dst_i];
        }
    }
    else if(global_src_i < m)
    {
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        for(Index new_j = 0; new_j < BLOCK_COL; new_j += BLOCK_COL_STEP)
        {
            if(global_src_j+new_j >= n)
            {
                break;
            }
            dst[global_src_j+new_j+global_src_i*n] =
                 T{alpha * Y{src[global_src_i+(global_src_j+new_j)*m]}};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Scalar alpha, const T *src,
        T *dst)
    noexcept
//! Transpose buffers on CPU
/*! dst[i,j] = alpha * src[j,i]
 *
 * @param[in] m: Number of rows of src and columns of dst
 * @param[in] n: Number of columns of src and rows of dst
 * @param[in] alpha: Scalar multiplier
 * @param[in] src: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(256);
    if(m < n)
    {
        if(m == 1)
        {
            scale::cuda<T>(stream, m*n, alpha, src, dst);
        }
        else if(m < 4)
        {
            dim3 blocks(((m+1)/2) * ((n+255)/256));
            (cuda_kernel<T, 2, 256, 2>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 8)
        {
            dim3 blocks(((m+3)/4) * ((n+255)/256));
            (cuda_kernel<T, 4, 256, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 16)
        {
            dim3 blocks(((m+7)/8) * ((n+127)/128));
            (cuda_kernel<T, 8, 128, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(m < 32)
        {
            dim3 blocks(((m+15)/16) * ((n+63)/64));
            (cuda_kernel<T, 16, 64, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
    }
    else
    {
        if(n == 1)
        {
            scale::cuda<T>(stream, m*n, alpha, src, dst);
        }
        else if(n < 4)
        {
            dim3 blocks(((m+255)/256) * ((n+1)/2));
            (cuda_kernel<T, 256, 2, 2>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 8)
        {
            dim3 blocks(((m+255)/256) * ((n+3)/4));
            (cuda_kernel<T, 256, 4, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 16)
        {
            dim3 blocks(((m+127)/128) * ((n+7)/8));
            (cuda_kernel<T, 128, 8, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else if(n < 32)
        {
            dim3 blocks(((m+63)/64) * ((n+15)/16));
            (cuda_kernel<T, 64, 16, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
        else
        {
            dim3 blocks(((m+31)/32) * ((n+31)/32));
            (cuda_kernel<T, 32, 32, 4>)<<<blocks, threads, 0, stream>>>(m, n,
                    alpha, src, dst);
        }
    }
}

//! Transpose an nr-by-nc sub-block of an m-by-n buffer on CUDA
/*! Performs the following operation for di in [0, nr) and dj in [0, nc):
 *      dst[di*n+dj] = alpha * src[di+dj*m]
 *
 * src and dst are the sub-block's base pointers (its (0, 0) corner in
 * the whole m-by-n matrix); the sub-block KEEPS the whole matrix's
 * strides - the src columns stay m elements apart, the dst rows n
 * elements apart. The chunked NNHaul wrapper drives this entry with
 * sub-blocks whose covered src column segments and whose covered dst
 * flat range each lie inside single device chunks; the whole-buffer
 * cuda() above is the (m, n) case. Internally shared-memory tiled
 * like the whole-buffer kernel: a 32x(32+1) tile with the load phase
 * coalesced along the contiguous src axis (the di axis) and the store
 * phase coalesced along the contiguous dst axis (the dj axis).
 * Elementwise and deterministic: any piece factorization is bitwise
 * the whole-buffer call.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of the whole src buffer.
 * @param[in] n: Size of the last mode of the whole src buffer.
 * @param[in] nr: Number of rows of the sub-block.
 * @param[in] nc: Number of columns of the sub-block.
 * @param[in] alpha: Scalar factor for src.
 * @param[in] src: Base pointer of the sub-block (m column stride).
 * @param[inout] dst: Base pointer of the sub-block (n row stride).
 * */
template<typename T, int TILE>
static __global__
void panel_cuda_kernel(Index m, Index n, Index nr, Index nc, Scalar alpha_,
    const T *src, T *dst)
{
    __shared__ T tile[TILE][TILE+1];
    Index const bx = blockIdx.x;
    Index const by = blockIdx.y;
    Index const tx = threadIdx.x;
    Index const ty = threadIdx.y;
    Index const rel_i = by*TILE + tx;
    Index const rel_j = bx*TILE + ty;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    // Load phase: consecutive threads read consecutive src elements
    // (the di axis is contiguous in src).
    if(rel_i < nr && rel_j < nc)
    {
        tile[tx][ty] = T{alpha * Y{src[rel_i + rel_j*m]}};
    }
    __syncthreads();
    // Store phase: consecutive threads write consecutive dst elements
    // (the dj axis is contiguous in dst); the value comes from the
    // transposed tile cell.
    if(by*TILE + ty < nr && bx*TILE + tx < nc)
    {
        dst[(by*TILE+ty)*n + bx*TILE+tx] = tile[ty][tx];
    }
}

template<typename T>
void panel(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const T* src, T* dst)
    noexcept
{
    if(nr <= 0 || nc <= 0)
    {
        return;
    }
    constexpr int TILE = 32;
    dim3 threads(TILE, TILE);
    dim3 blocks((nc+TILE-1)/TILE, (nr+TILE-1)/TILE);
    (panel_cuda_kernel<T, TILE>)<<<blocks, threads, 0, stream>>>(m, n, nr,
            nc, alpha, src, dst);
}

//! Chunk descriptors for the one-launch chunked transpose (see cuda.hh).
template<typename T>
struct ChunkTables
{
    const T* const* src; //!< device-ring view of the src chunk-pointer array
    T* const* dst; //!< device-ring view of the dst chunk-pointer array
    Index nsrc;
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    Index m;
    Index n;
    Scalar alpha;
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a src flat element offset (power-of-two shift when the
//! chunk size is a power of two).
template<typename T>
static __device__ __forceinline__ Index src_chunk_of(ChunkTables<T> const &t,
    long f)
{
    return t.shift_src >= 0 ? Index(f >> t.shift_src)
                            : Index(f / long(t.elems_src));
}

//! Chunk index of a dst flat element offset.
template<typename T>
static __device__ __forceinline__ Index dst_chunk_of(ChunkTables<T> const &t,
    long g)
{
    return t.shift_dst >= 0 ? Index(g >> t.shift_dst)
                            : Index(g / long(t.elems_dst));
}

//! Chunk-aware transpose tile kernel: the same shared-memory tiling and
//! launch-geometry family as cuda_kernel above (the caller picks the
//! <BR, BC, BL> configuration from the aspect ratio like the whole-buffer
//! entry does), but each operand is addressed through its chunk table.
//! The tables are read from the device chunk-pointer ring (a block-uniform
//! read); in the healthy regime a tile's flat range spans at most two
//! chunks, so the two candidate base pointers are hoisted once per block
//! (a block-uniform constant read) and only a per-element compare picks
//! between them. Elementwise and deterministic: bitwise the whole-buffer
//! call.
template<typename T, int BR, int BC, int BL>
static __global__
void chunked_cuda_kernel(ChunkTables<T> t)
{
    Index const m = t.m;
    Index const n = t.n;
    Index const src_i = threadIdx.x % BR;
    Index const src_j = threadIdx.x / BR;
    Index const grid_row = (m + BR - 1) / BR;
    Index const bi = blockIdx.x % grid_row;
    Index const bj = blockIdx.x / grid_row;
    Index const gi = src_i + bi * BR;
    Index const gj = src_j + bj * BC;
    constexpr int STEP = BC / BL;
    if((bi + 1) * BR <= m && (bj + 1) * BC <= n)
    {
        __shared__ T blk[BR][BC + 1];
        // The whole tile's flat range spans at most two chunks in the
        // regime this path serves; hoist the two candidate base pointers
        // per block (uniform, a broadcast constant read) and select them
        // per element. When a tile does span more than two chunks (a very
        // narrow chunk class), re-hoist per column step instead, whose
        // flat range is short enough to cross at most one boundary.
        Index const dst_i = threadIdx.x % BC;
        Index const dst_j = threadIdx.x / BC;
        constexpr int RSTEP = BR / BL;
        long const fmin = long(bi) * BR + long(bj) * BC * long(m);
        long const fmax = long(bi * BR + BR - 1)
            + long(bj * BC + BC - 1) * long(m);
        Index const cs0 = src_chunk_of(t, fmin);
        Index const cs1 = src_chunk_of(t, fmax);
        if(cs1 <= cs0 + 1)
        {
            T const *sp0 = t.src[cs0];
            T const *sp1 = cs1 > cs0 ? t.src[cs1] : sp0;
            long const bond = long(cs0 + 1) * long(t.elems_src);
            long const off0 = long(cs0) * long(t.elems_src);
            bool const two = cs1 > cs0;
            for(int k = 0; k < BC; k += STEP)
            {
                long const f = long(gi) + long(gj + k) * long(m);
                T const *sp = sp0;
                long off = f - off0;
                if(two && f >= bond)
                {
                    sp = sp1;
                    off = f - bond;
                }
                blk[src_i][src_j + k] = T{t.alpha
                    * typename T::repr_t{sp[off]}};
            }
        }
        else
        {
            for(int k = 0; k < BC; k += STEP)
            {
                long const f_lo = long(bi) * BR
                    + long(bj) * BC * long(m) + long(k) * long(m);
                long const f_hi = long(bi * BR + BR - 1)
                    + long(bj * BC + k + STEP - 1) * long(m);
                Index const k0 = src_chunk_of(t, f_lo);
                Index const k1 = src_chunk_of(t, f_hi);
                T const *p0 = t.src[k0];
                T const *p1 = k1 > k0 ? t.src[k1] : p0;
                long const bond = long(k0 + 1) * long(t.elems_src);
                long const off0 = long(k0) * long(t.elems_src);
                bool const two = k1 > k0;
                long const f = long(gi) + long(gj + k) * long(m);
                T const *sp = p0;
                long off = f - off0;
                if(two && f >= bond)
                {
                    sp = p1;
                    off = f - bond;
                }
                // Still multi-chunk (tiny class): index the table.
                if(k1 > k0 + 1)
                {
                    Index const c = src_chunk_of(t, f);
                    sp = t.src[c];
                    off = f - long(c) * long(t.elems_src);
                }
                blk[src_i][src_j + k] = T{t.alpha
                    * typename T::repr_t{sp[off]}};
            }
        }
        long const gmin = long(bj) * BC + long(bi) * BR * long(n);
        long const gmax = long(bj * BC + BC - 1)
            + long(bi * BR + BR - 1) * long(n);
        Index const cd0 = dst_chunk_of(t, gmin);
        Index const cd1 = dst_chunk_of(t, gmax);
        T *dp0 = t.dst[cd0];
        T *dp1 = cd1 > cd0 ? t.dst[cd1] : dp0;
        long const bond_d = long(cd0 + 1) * long(t.elems_dst);
        long const doff0 = long(cd0) * long(t.elems_dst);
        bool const d_two = cd1 > cd0;
        bool const d_multi = cd1 > cd0 + 1;
        __syncthreads();
        for(int k = 0; k < BR; k += RSTEP)
        {
            long const g = long(dst_i + bj * BC)
                + long(dst_j + bi * BR + k) * long(n);
            T *dp = dp0;
            long off = g - doff0;
            if(d_multi)
            {
                Index const c = dst_chunk_of(t, g);
                dp = t.dst[c];
                off = g - long(c) * long(t.elems_dst);
            }
            else if(d_two && g >= bond_d)
            {
                dp = dp1;
                off = g - bond_d;
            }
            dp[off] = blk[dst_j + k][dst_i];
        }
    }
    else
    {
        // Edge tiles: bounds-checked elementwise chunk-mapped transpose.
        for(Index idx = threadIdx.x; idx < BR * BC; idx += blockDim.x)
        {
            Index const ii = idx % BR;
            Index const jj = idx / BR;
            Index const di = bi * BR + ii;
            Index const dj = bj * BC + jj;
            if(di >= m || dj >= n)
            {
                continue;
            }
            long const f = long(di) + long(dj) * long(m);
            Index const cs = src_chunk_of(t, f);
            long const g = long(di) * long(n) + long(dj);
            Index const cd = dst_chunk_of(t, g);
            t.dst[cd][g - long(cd) * long(t.elems_dst)] = T{t.alpha
                * typename T::repr_t{t.src[cs]
                    [f - long(cs) * long(t.elems_src)]}};
        }
    }
}

//! Launch chunked_cuda_kernel with the whole-buffer entry's geometry.
template<typename T, int BR, int BC, int BL>
static void launch_chunked(cudaStream_t stream, ChunkTables<T> const &t)
{
    dim3 threads(BR * BC / BL);
    Index const grid_row = (t.m + BR - 1) / BR;
    Index const grid_col = (t.n + BC - 1) / BC;
    dim3 blocks(grid_row * grid_col);
    (chunked_cuda_kernel<T, BR, BC, BL>)<<<blocks, threads, 0, stream>>>(t);
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const T* const* src_chunks, Index nsrc,
    T* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.nsrc = nsrc;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.m = m;
    t.n = n;
    t.alpha = alpha;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_transpose");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_transpose");
    // Same aspect-ratio geometry selection as the whole-buffer entry: the
    // narrow operand's extent picks the tile shape, so a 12-wide matrix
    // gets 128x8 tiles with 128-wide coalescing instead of degenerate
    // 32x32 blocks whose every tile runs off the edge.
    if(m < n)
    {
        if(m >= 32)
        {
            launch_chunked<T, 32, 32, 4>(stream, t);
        }
        else
        {
            // Narrow first operand: pick BR dividing m so every row block
            // is a full tile (the whole-buffer entry accepts a ragged
            // last row block and lets it take the scalar edge path; at
            // these chunk granularities that costs more than a smaller
            // tile does). BR then fixes BC = 256*BL/BR.
            Index br = 16;
            while(br > 2 && m % br != 0)
            {
                br /= 2;
            }
            if(br < 2 || m % br != 0)
            {
                br = 2;
            }
            if(br >= 16)
            {
                launch_chunked<T, 16, 64, 4>(stream, t);
            }
            else if(br >= 8)
            {
                launch_chunked<T, 8, 128, 4>(stream, t);
            }
            else if(br >= 4)
            {
                launch_chunked<T, 4, 256, 4>(stream, t);
            }
            else
            {
                launch_chunked<T, 2, 256, 2>(stream, t);
            }
        }
    }
    else
    {
        if(n < 4)
        {
            launch_chunked<T, 256, 2, 2>(stream, t);
        }
        else if(n < 8)
        {
            launch_chunked<T, 256, 4, 4>(stream, t);
        }
        else if(n < 16)
        {
            // The whole-buffer entry's fixed 128x8 tile leaves a ragged
            // last column block whenever 8 does not divide n (n=12 ->
            // columns 8..11) and that block pays the kernel's
            // per-element edge path. Prefer a BC that divides n so every
            // tile is full and the store phase is linear in the
            // destination's contiguous axis. Measured (GPT-2 m=65536,
            // n=12, 256 KiB chunks): 64x12 is ~0.65-0.75x the contiguous
            // kernel, 256x4 ~0.8x, fixed 128x8 1.0-1.06x.
            if(n % 12 == 0)
            {
                launch_chunked<T, 64, 12, 4>(stream, t);
            }
            else if(n % 8 == 0)
            {
                launch_chunked<T, 128, 8, 4>(stream, t);
            }
            else if(n % 4 == 0)
            {
                launch_chunked<T, 256, 4, 4>(stream, t);
            }
            else if(n % 2 == 0)
            {
                launch_chunked<T, 256, 2, 2>(stream, t);
            }
            else
            {
                launch_chunked<T, 128, 8, 4>(stream, t);
            }
        }
        else if(n < 32)
        {
            launch_chunked<T, 64, 16, 4>(stream, t);
        }
        else
        {
            launch_chunked<T, 32, 32, 4>(stream, t);
        }
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp32_t* src, fp32_t* dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp64_t* src, fp64_t* dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const bf16_t* src, bf16_t* dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
        const fp16_t* src, fp16_t* dst)
    noexcept;

template
void panel<fp32_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp32_t* src, fp32_t* dst)
    noexcept;

template
void panel<fp64_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp64_t* src, fp64_t* dst)
    noexcept;

template
void panel<bf16_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const bf16_t* src, bf16_t* dst)
    noexcept;

template
void panel<fp16_t>(cudaStream_t stream, Index m, Index n, Index nr, Index nc,
    Scalar alpha, const fp16_t* src, fp16_t* dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const fp32_t* const* src_chunks, Index nsrc,
    fp32_t* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const fp64_t* const* src_chunks, Index nsrc,
    fp64_t* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const bf16_t* const* src_chunks, Index nsrc,
    bf16_t* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Scalar alpha,
    const fp16_t* const* src_chunks, Index nsrc,
    fp16_t* const* dst_chunks, Index ndst,
    Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::tranpose
