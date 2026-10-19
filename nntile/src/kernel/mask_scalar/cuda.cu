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

//! Chunk descriptors for the one-launch chunked mask_scalar (see cuda.hh).
template<typename T>
struct ChunkTables
{
    T *const *data; //!< device-ring view of the data chunk-pointer array
    const bool_t *const *mask; //!< device-ring view of the mask table
    Index ndata;
    Index nmask;
    Index elems_data; //!< elements in a full data chunk
    Index elems_mask; //!< elements in a full mask chunk
    Index nrows;
    Index ncols;
    Scalar val;
    int shift_data; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_mask; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk-aware mask-scalar kernel: exactly the classic cuda_kernel2
//! geometry ((i, j) work assignment, shared row-mask load, LOOP column
//! steps), but each operand is addressed through its chunk table. Piece 9
//! (rule 10): the tables are read straight from the device chunk-pointer
//! ring by EVERY thread that needs an entry -- no shared-memory staging,
//! no table __syncthreads (the piece-7 staging predates the rule-7
//! compile-time shifts and was a workaround for runtime shifts). The
//! per-operand chunk shifts stay COMPILE-TIME template constants (the
//! DATA1/MASK1 flags pick the sentinel 62 for a 1-chunk operand -- no
//! chunk arithmetic at all for it -- or the rule-7 class shift), so the
//! store loop's shift/mask lower to immediates. U32 selects the 32-bit
//! incremental column walk (valid whenever the whole tile spans fewer
//! than 2^32 elements, host-checked); the U32=false instantiation keeps
//! 64-bit flat offsets for arbitrarily large tiles (the ring's no-cap
//! contract). Writes are idempotent val assignments and the pointer
//! VALUES are the staged form's own, so the result is bitwise the staged
//! form's and the whole-buffer cuda() call's.
template<typename T, typename Tab, int BLOCK_ROW, int BLOCK_COL, int LOOP,
    bool DATA1, bool MASK1, bool U32>
static __global__
void chunked_cuda_kernel(Tab t)
{
    // Rule-7 restatement: the multi-chunk full-chunk shift IS the class
    // shift (chunk_shift_of validated it host-side and aborts otherwise);
    // a 1-chunk operand uses the sentinel. Compile-time constants, so the
    // store loop's shift/mask lower to immediates.
    constexpr int SHIFT_DATA = DATA1 ? nntile::kernel::kChunkSingleShift
        : nntile::kernel::chunk_log2(
            nntile::kernel::kChunkClassBytes / sizeof(T));
    constexpr int SHIFT_MASK = MASK1 ? nntile::kernel::kChunkSingleShift
        : nntile::kernel::chunk_log2(
            nntile::kernel::kChunkClassBytes / sizeof(nntile::bool_t));
    Index griddim_row = (t.nrows+BLOCK_ROW-1) / BLOCK_ROW;
    Index block_i = blockIdx.x % griddim_row;
    Index block_j = blockIdx.x / griddim_row;
    Index i = threadIdx.x % BLOCK_ROW;
    Index j = threadIdx.x / BLOCK_ROW;
    Index global_i = block_i*BLOCK_ROW + i;
    using Y = typename T::repr_t;
    const T val = static_cast<T>(static_cast<Y>(t.val));
    __shared__ bool mask_block[BLOCK_ROW];
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    if(threadIdx.x < BLOCK_ROW and global_i < t.nrows)
    {
        // Per-thread direct ring read (rule 10); a 1-chunk mask operand
        // resolves to table slot 0 with no chunk arithmetic (shift 62).
        unsigned const u_i = unsigned(global_i);
        mask_block[threadIdx.x] = reinterpret_cast<const bool *>(
            t.mask[u_i >> SHIFT_MASK])[u_i & ((1u << SHIFT_MASK) - 1u)];
    }
    __syncthreads();
    if(global_i < t.nrows)
    {
        // The row's mask bit was resolved once per block above (the classic
        // kernel re-reads it from global memory here); use the shared copy
        // so the chunk-table lookup happens once per row per block.
        if(!mask_block[i])
        {
            if(U32)
            {
                // 32-bit incremental walk: f advances by the constant
                // column step (no per-store 64-bit multiply), the chunk
                // index is an immediate shift of a 32-bit value, and the
                // data chunk pointer is loaded straight from the ring
                // table per store (L1-resident, a handful of entries).
                unsigned const nrows32 = unsigned(t.nrows);
                unsigned const step = unsigned(BLOCK_COL_STEP) * nrows32;
                unsigned const jend = ::min(
                    unsigned((block_j+1)*BLOCK_COL), unsigned(t.ncols));
                unsigned j32 = unsigned(block_j*BLOCK_COL+j);
                unsigned f = j32*nrows32 + unsigned(global_i);
                for(; j32 < jend; j32 += BLOCK_COL_STEP, f += step)
                {
                    t.data[f >> SHIFT_DATA]
                        [f & ((1u << SHIFT_DATA) - 1u)] = val;
                }
            }
            else
            {
                // 64-bit fallback for tiles beyond 2^32 elements; same
                // incremental walk, same direct ring reads.
                long const nrows64 = long(t.nrows);
                long const step = long(BLOCK_COL_STEP) * nrows64;
                long const jend = ::min(long((block_j+1)*BLOCK_COL),
                    long(t.ncols));
                long j64 = long(block_j*BLOCK_COL+j);
                long f = j64*nrows64 + long(global_i);
                for(; j64 < jend; j64 += BLOCK_COL_STEP, f += step)
                {
                    t.data[f >> SHIFT_DATA]
                        [f & ((1LL << SHIFT_DATA) - 1)] = val;
                }
            }
        }
    }
}

//! Launch the chunk-aware kernel with the whole-buffer entry's geometry.
//! Rule 10: one instantiation cross on (DATA1, MASK1, U32) -- all direct
//! ring reads; U32 is host-guaranteed by nrows*ncols < 2^32.
template<typename T, typename Tab>
static void launch_mask_chunked(cudaStream_t stream, Tab const &t)
{
    dim3 threads(256);
    dim3 blocks(((t.nrows+127)/128) * ((t.ncols+31)/32));
    bool const data1 = t.ndata <= 1;
    bool const mask1 = t.nmask <= 1;
    bool const u32 = static_cast<unsigned long long>(t.nrows)
        *static_cast<unsigned long long>(t.ncols) < (1ull << 32);
#define NNHAUL_MASK_LAUNCH(D1, M1, U) \
    (chunked_cuda_kernel<T, Tab, 128, 32, 16, D1, M1, U>) \
        <<<blocks, threads, 0, stream>>>(t)
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
    ChunkTables<T> t;
    t.data = data_chunks;
    t.mask = mask_chunks;
    t.ndata = ndata;
    t.nmask = nmask;
    t.elems_data = elems_data;
    t.elems_mask = elems_mask;
    t.nrows = nrows;
    t.ncols = ncols;
    t.val = val;
    t.shift_data = chunk_shift_of<T>(t.ndata, elems_data, "nntile_mask_scalar");
    t.shift_mask = chunk_shift_of<::nntile::bool_t>(t.nmask, elems_mask, "nntile_mask_scalar");
    launch_mask_chunked<T, ChunkTables<T>>(stream, t);
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
