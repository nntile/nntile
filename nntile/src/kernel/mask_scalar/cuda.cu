/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/mask_scalar/cuda.cu
 * Mask operation with scalar on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/mask_scalar/cuda.hh"
#include "nntile/kernel/cuda.hh"

#include <cstdio>

namespace nntile::kernel::mask_scalar
{

template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP>
static __global__
void cuda_kernel2(Index nrows, Index ncols, const bool *mask, Scalar val_,
        T *data)
{
    Index griddim_row = (nrows+BLOCK_ROW-1) / BLOCK_ROW;
    Index block_i = blockIdx.x % griddim_row;
    Index block_j = blockIdx.x / griddim_row;
    Index i = threadIdx.x % BLOCK_ROW;
    Index j = threadIdx.x / BLOCK_ROW;
    Index global_i = block_i*BLOCK_ROW + i;
    using Y = typename T::repr_t;
    const T val = static_cast<T>(static_cast<Y>(val_));
    __shared__ bool mask_block[BLOCK_ROW];
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    if(threadIdx.x < BLOCK_ROW and global_i < nrows)
    {
        mask_block[threadIdx.x] = mask[global_i];
    }
    __syncthreads();
    if(global_i < nrows)
    {
        if(!mask[global_i])
        {
            for(Index global_j = block_j*BLOCK_COL+j;
                    global_j < ::min((block_j+1)*BLOCK_COL, ncols);
                    global_j += BLOCK_COL_STEP)
            {
                data[global_j*nrows+global_i] = val;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nrows, Index ncols, const bool_t *mask_,
        Scalar val, T *data)
    noexcept
//! Set certain matrix entries to a given value by mask on CUDA
/*! Does the following operation:
 *      if(!mask[i]) data[i,:] = val
 *
 * @params[in] nrows: Number of rows of data
 * @params[in] ncols: Number of columns of data
 * @params[in] mask_: buffer with mask values with nrows entries
 * @params[in] val: value to set if mask element is false
 * @params[inout] data: nrows by ncols matrix, whose elements are updated
 * */
{
    dim3 threads(256);
    using B = typename CUDAComputeType<bool_t>::value;
    auto mask = reinterpret_cast<const B *>(mask_);
    dim3 blocks(((nrows+127)/128) * ((ncols+31)/32));
    (cuda_kernel2<T, 128, 32, 16>)<<<blocks, threads, 0, stream>>>(nrows,
            ncols, mask, val, data);
}

//! Chunk-aware mask-scalar kernel: EXACTLY the classic cuda_kernel2's
//! generated code shape, with the two operands addressed through their
//! device-ring chunk tables. The parity-relevant facts (all verified in
//! SASS): (1) the classic kernel stages the row-mask byte into shared
//! memory but never READS the staged value (its branch re-reads
//! mask[global_i] from global), so nvcc's dead-store elimination
//! deletes the staging load/store AND the shared array -- only a
//! vestigial __syncthreads survives -- and the classic hot path is one
//! mask byte load per thread. The previous chunked kernel READ the
//! staged byte (mask_block[i]), which kept the staging alive: a
//! dependent table load -> byte load -> STS chain, the block-wide
//! barrier, and an LDS on every thread's exit path. On the attention
//! bench (an all-true 1 MiB mask, so every thread takes the
//! keep-the-row exit) that live barrier + double-indirect load was the
//! entire ~13% parity gap: the kernel is latency-bound on the mask
//! reads, and the chunked form paid table-load latency SERIALLY before
//! the byte load plus a barrier before any thread could exit. This
//! kernel mirrors the classic's post-DSE form: NO shared array, NO
//! barrier, ONE mask byte load per thread.
//! (2) The mask operand's virtual contiguous base (the pointer q with
//! the element at flat offset u living at q[u]) is resolved ONCE per
//! thread and, for a 1-chunk mask (MASK1), the table-slot-0 load is
//! issued as the kernel's FIRST statement -- its address is the table
//! parameter itself, so the load leaves the core immediately and its
//! latency hides under the grid-index division math (a runtime
//! 64-bit division both kernels pay identically). A multi-chunk mask
//! resolves its slot from global_i after the bounds check (the slot
//! index of an out-of-range row could read past the table, so the
//! resolution stays guarded exactly like the classic's loads).
//! (3) The store walk keeps the classic (i, j) geometry and column
//! stepping; each RUN of columns that lands inside one data chunk uses
//! that chunk's virtual contiguous base (a straight q[f] store, the
//! classic's own addressing once the base is in a register -- no
//! per-store table traffic), and a run that would cross a chunk
//! boundary rebases through the table (rule 7: shift/mask on the
//! compile-time class shift, or the sentinel 62 for a 1-chunk operand,
//! which the compiler folds to an unconditional single-chunk fast
//! path). The rebase load for the NEXT store is issued only when that
//! store will actually happen, so the flat offset handed to the table
//! is always a real in-buffer element offset (no out-of-range table
//! slot is ever read, even when the last column step overshoots the
//! buffer). U32 selects the 32-bit walk (host-guaranteed by
//! nrows*ncols < 2^32); U32=false keeps 64-bit flat offsets for
//! arbitrarily large tiles (the ring's no-cap contract). The stores
//! are the classic's idempotent conditional val assignments to the
//! classic's exact element set in the classic's exact order, so the
//! result is bitwise the whole-buffer cuda() call's, in-place dst
//! aliasing included.
template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP,
    bool DATA1, bool MASK1, bool U32>
static __global__
void chunked_cuda_kernel(Index nrows, Index ncols, Scalar val_,
    T *const *data_tab, const bool_t *const *mask_tab)
{
    // Rule-7 restatement: the multi-chunk full-chunk shift IS the class
    // shift (chunk_shift_of validated it host-side and aborts otherwise);
    // a 1-chunk operand uses the sentinel. Compile-time constants, so the
    // chunk divmod lowers to shift/mask immediates.
    constexpr int SHIFT_DATA = DATA1 ? nntile::kernel::kChunkSingleShift
        : nntile::kernel::chunk_log2(
            nntile::kernel::kChunkClassBytes / sizeof(T));
    constexpr int SHIFT_MASK = MASK1 ? nntile::kernel::kChunkSingleShift
        : nntile::kernel::chunk_log2(
            nntile::kernel::kChunkClassBytes / sizeof(nntile::bool_t));
    using Y = typename T::repr_t;
    const T val = static_cast<T>(static_cast<Y>(val_));
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    // Single-chunk mask: the base is table slot 0 and its address is the
    // table parameter -- issue the load FIRST so its latency overlaps the
    // grid-index division below (a warp-uniform, L1-resident load).
    // nvcc schedules an ordinary table load just before its use, which
    // puts the load's full latency on the exit path of every row-kept
    // thread (this kernel is latency-bound on the mask reads), so the
    // early issue is pinned with a volatile inline-PTX ld.global.u64 --
    // the SAME 8-byte load the portable form issues, only hoisted above
    // the division math.
    const bool *mq;
    if constexpr(MASK1)
    {
        unsigned long long mq_bits;
        asm volatile("ld.global.u64 %0, [%1];"
            : "=l"(mq_bits)
            : "l"(mask_tab)
            : "memory");
        mq = reinterpret_cast<const bool *>(mq_bits);
    }
    Index griddim_row = (nrows+BLOCK_ROW-1) / BLOCK_ROW;
    Index block_i = blockIdx.x % griddim_row;
    Index block_j = blockIdx.x / griddim_row;
    Index i = threadIdx.x % BLOCK_ROW;
    Index j = threadIdx.x / BLOCK_ROW;
    Index global_i = block_i*BLOCK_ROW + i;
    if(global_i < nrows)
    {
        // Multi-chunk mask: resolve the row's chunk after the bounds
        // check (an out-of-range row's slot could sit past the table).
        if constexpr(!MASK1)
        {
            long const c = long(global_i) >> SHIFT_MASK;
            mq = reinterpret_cast<const bool *>(mask_tab[c])
                - c*(std::size_t(1) << SHIFT_MASK);
        }
        // The classic kernel's branch load (its staging copy was dead
        // code); one byte load per thread through the resolved base.
        if(!mq[global_i])
        {
            if constexpr(U32)
            {
                // 32-bit incremental walk.
                unsigned const nrows32 = unsigned(nrows);
                unsigned const step = unsigned(BLOCK_COL_STEP) * nrows32;
                unsigned jend = unsigned(block_j+1)*unsigned(BLOCK_COL);
                if(jend > unsigned(ncols))
                {
                    jend = unsigned(ncols);
                }
                unsigned j32 = unsigned(block_j*BLOCK_COL+j);
                unsigned f = j32*nrows32 + unsigned(global_i);
                if(j32 < jend)
                {
                    // Store count of this thread's column walk (>= 1).
                    unsigned const nsteps =
                        (jend-j32+unsigned(BLOCK_COL_STEP)-1)
                        / unsigned(BLOCK_COL_STEP);
                    unsigned const fhi = f + (nsteps-1)*step;
                    unsigned const c0 = f >> SHIFT_DATA;
                    if(DATA1 or (fhi >> SHIFT_DATA) == c0)
                    {
                        // Whole walk inside one chunk (always for a
                        // 1-chunk data operand): the classic's straight
                        // q[f] store loop off a virtual contiguous base.
                        T *q = DATA1
                            ? data_tab[0]
                            : data_tab[c0]
                                - c0*(std::size_t(1) << SHIFT_DATA);
                        for(; j32 < jend; j32 += BLOCK_COL_STEP, f += step)
                        {
                            q[f] = val;
                        }
                    }
                    else
                    {
                        // Per-RUN rebase walk: off advances by the
                        // constant column step and the table is touched
                        // only when the next store crosses a chunk
                        // boundary, and only when a next store exists.
                        unsigned off = f - (c0 << SHIFT_DATA);
                        T *q = data_tab[c0];
                        for(; j32 < jend;)
                        {
                            q[off] = val;
                            j32 += BLOCK_COL_STEP;
                            if(j32 >= jend)
                            {
                                break;
                            }
                            f += step;
                            off += step;
                            if(off >> SHIFT_DATA != 0)
                            {
                                unsigned const c = f >> SHIFT_DATA;
                                q = data_tab[c];
                                off = f - (c << SHIFT_DATA);
                            }
                        }
                    }
                }
            }
            else
            {
                // 64-bit walk for tiles beyond 2^32 elements.
                long const nrows64 = long(nrows);
                long const step = long(BLOCK_COL_STEP) * nrows64;
                long jend = long(block_j+1)*long(BLOCK_COL);
                if(jend > long(ncols))
                {
                    jend = long(ncols);
                }
                long j64 = long(block_j*BLOCK_COL+j);
                long f = j64*nrows64 + long(global_i);
                if(j64 < jend)
                {
                    long const nsteps =
                        (jend-j64+long(BLOCK_COL_STEP)-1)
                        / long(BLOCK_COL_STEP);
                    long const fhi = f + (nsteps-1)*step;
                    long const c0 = f >> SHIFT_DATA;
                    if(DATA1 or (fhi >> SHIFT_DATA) == c0)
                    {
                        T *q = DATA1
                            ? data_tab[0]
                            : data_tab[c0]
                                - c0*(std::size_t(1) << SHIFT_DATA);
                        for(; j64 < jend; j64 += BLOCK_COL_STEP, f += step)
                        {
                            q[f] = val;
                        }
                    }
                    else
                    {
                        long off = f - (c0 << SHIFT_DATA);
                        T *q = data_tab[c0];
                        for(; j64 < jend;)
                        {
                            q[off] = val;
                            j64 += BLOCK_COL_STEP;
                            if(j64 >= jend)
                            {
                                break;
                            }
                            f += step;
                            off += step;
                            if(off >> SHIFT_DATA != 0)
                            {
                                long const c = f >> SHIFT_DATA;
                                q = data_tab[c];
                                off = f - (c << SHIFT_DATA);
                            }
                        }
                    }
                }
            }
        }
    }
}

//! Launch the chunk-aware kernel with the whole-buffer entry's geometry.
//! Rule 10: one instantiation cross on (DATA1, MASK1, U32); U32 is
//! host-guaranteed by nrows*ncols < 2^32. The rule-7 validation
//! (chunk_shift_of) runs host-side and aborts loudly on a non-class
//! multi-chunk operand.
template<typename T>
static void launch_mask_chunked(cudaStream_t stream, Index nrows,
    Index ncols, Scalar val, T *const *data_chunks, Index ndata,
    Index elems_data, const bool_t *const *mask_chunks, Index nmask,
    Index elems_mask)
{
    (void) chunk_shift_of<T>(std::size_t(ndata), std::size_t(elems_data),
        "nntile_mask_scalar");
    (void) chunk_shift_of<::nntile::bool_t>(std::size_t(nmask),
        std::size_t(elems_mask), "nntile_mask_scalar");
    dim3 threads(256);
    dim3 blocks(((nrows+127)/128) * ((ncols+31)/32));
    bool const data1 = ndata <= 1;
    bool const mask1 = nmask <= 1;
    bool const u32 = static_cast<unsigned long long>(nrows)
        *static_cast<unsigned long long>(ncols) < (1ull << 32);
#define NNHAUL_MASK_LAUNCH(D1, M1, U) \
    (chunked_cuda_kernel<T, 128, 32, 16, D1, M1, U>) \
        <<<blocks, threads, 0, stream>>>(nrows, ncols, val, data_chunks, \
            mask_chunks)
    if(u32)
    {
        if(data1)
        {
            if(mask1) { NNHAUL_MASK_LAUNCH(true, true, true); }
            else { NNHAUL_MASK_LAUNCH(true, false, true); }
        }
        else
        {
            if(mask1) { NNHAUL_MASK_LAUNCH(false, true, true); }
            else { NNHAUL_MASK_LAUNCH(false, false, true); }
        }
    }
    else
    {
        if(data1)
        {
            if(mask1) { NNHAUL_MASK_LAUNCH(true, true, false); }
            else { NNHAUL_MASK_LAUNCH(true, false, false); }
        }
        else
        {
            if(mask1) { NNHAUL_MASK_LAUNCH(false, true, false); }
            else { NNHAUL_MASK_LAUNCH(false, false, false); }
        }
    }
#undef NNHAUL_MASK_LAUNCH
}

//! Chunked mask_scalar with the base-pointer tables in the device
//! chunk-pointer ring (no chunk-count cap; the caller supplies a device view).
template<typename T>
void chunked(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
    T *const *data_chunks, Index ndata, Index elems_data,
    const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept
{
    if(nrows <= 0 || ncols <= 0 || ndata <= 0 || nmask <= 0)
    {
        return;
    }
    launch_mask_chunked<T>(stream, nrows, ncols, val, data_chunks, ndata,
        elems_data, mask_chunks, nmask, elems_mask);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp16_t *data)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
        fp32_t *const *data_chunks, Index ndata, Index elems_data,
        const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
        fp64_t *const *data_chunks, Index ndata, Index elems_data,
        const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
        bf16_t *const *data_chunks, Index ndata, Index elems_data,
        const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nrows, Index ncols, Scalar val,
        fp16_t *const *data_chunks, Index ndata, Index elems_data,
        const bool_t *const *mask_chunks, Index nmask, Index elems_mask)
    noexcept;

} // namespace nntile::kernel::mask_scalar
