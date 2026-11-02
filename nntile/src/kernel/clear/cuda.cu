/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech, Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/clear/cuda.cu
 * The ONE-launch chunked clear (a single kernel: the chunk table is
 * device-resident, so the host cannot issue per-chunk cudaMemsetAsync
 * without a synchronizing copy).
 *
 * @version 1.4.0
 * */

#include "nntile/kernel/clear/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::clear
{

//! Chunk tables for the one-launch chunked clear (see cuda.hh).
struct ChunkTables
{
    unsigned char *const* data; //!< device-ring view of the data ptr array
    Index ndata;
    Index elems_data; //!< bytes in a full data chunk
    int shift_data; //!< class shift (multi-chunk) or the 62 sentinel
};

/*! Memset-shaped flat walk. Measured structure race on the 37-chunk
 * 147 MiB wte-grad shape and the 3-chunk 9 MiB mlp-w shape (RTX 2080 Ti,
 * .zcode/bench-kernel/live/laneB/p14/clr_micro.cu): within one store
 * instruction the 32 lanes must cover CONSECUTIVE 16-byte units (512B
 * per warp instruction, every 32-byte sector fully written) -- that is
 * the memset pattern; any per-thread-consecutive multi-store span
 * (32/64/128B per thread) halves sector fills per instruction and loses
 * 2-80% (128B/thread: 479us vs 268us memset), and loop/span forms with
 * per-unit branch chains lose another ~10%. Straight-line 2-unit ILP
 * ties the one-unit form on the big shape and costs ~4% on the
 * latency-sensitive 9 MiB shape, so the winner is the leanest possible
 * flat form: ONE unit per thread, one guard, one resolve, one store
 * (the old walk's 4x grid-stride unroll never fired at its sized grid
 * and its machinery measured ~free but the guard cost did not).
 * Remaining gap to the classic cudaMemsetAsync (268.3us): the driver's
 * fill runs on a faster path than any SM store sequence can reach -- an
 * SM kernel over the SAME contiguous buffer with ZERO table work floors
 * at 281.3us (540 GB/s vs 566 GB/s), so the last ~5% is not reachable
 * from device code. The clear size still ROUNDS UP to a 16-byte
 * multiple (the pool's chunk blocks and cudaMalloc's 256-byte
 * granularity absorb the <=15-byte overclear), so every unit is a full,
 * unconditional uint4 store. MULTI=false (single-chunk operand, the
 * sentinel shift 62) strips the chunk arithmetic to one broadcast table
 * read per thread. */
template<bool MULTI>
static __global__
void chunked_cuda_kernel(Index nbytes, ChunkTables t)
{
    long const units = (long(nbytes) + 15) / 16; // rounded up (padding)
    long const u = long(threadIdx.x) + long(blockIdx.x)*blockDim.x;
    if(u >= units)
    {
        return;
    }
    long const f = u * 16;
    if constexpr(MULTI)
    {
        Index const c = Index(f >> t.shift_data);
        *reinterpret_cast<uint4 *>(t.data[c]
            + (f - long(c) * long(t.elems_data))) = make_uint4(0, 0, 0, 0);
    }
    else
    {
        *reinterpret_cast<uint4 *>(t.data[0] + f) = make_uint4(0, 0, 0, 0);
    }
}

void chunked(cudaStream_t stream, Index nbytes,
        unsigned char* const* data_chunks, Index ndata, Index elems_data)
    noexcept
{
    if(nbytes <= 0 || ndata <= 0)
    {
        return;
    }
    ChunkTables t;
    t.data = data_chunks;
    t.ndata = ndata;
    t.elems_data = elems_data;
    t.shift_data = chunk_shift_of<unsigned char>(t.ndata, elems_data,
        "nntile_clear");
    Index const units = (nbytes + 15) / 16;
    dim3 blocks((units+255)/256), threads(256);
    if(t.ndata == 1)
    {
        (chunked_cuda_kernel<false>)<<<blocks, threads, 0, stream>>>(
            nbytes, t);
    }
    else
    {
        (chunked_cuda_kernel<true>)<<<blocks, threads, 0, stream>>>(
            nbytes, t);
    }
}

} // namespace nntile::kernel::clear
