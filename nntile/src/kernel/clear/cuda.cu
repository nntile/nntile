/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech, Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/clear/cuda.cu
 * The ONE-launch chunked clear (user directive 2026-10-19: a single
 * kernel, because per-chunk cudaMemsetAsync does not scale -- a 16 MiB
 * chunk is ~4 us of fill at H200's 3.6 TB/s, where launch overhead
 * dominates).
 *
 * @version 1.3.0
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
    int shift_data; //!< class shift (multi-chunk)
};

/*! Branch-free one-unit-per-touch kernel: the clear size ROUNDS UP to a
 * 16-byte multiple (the pool's chunk blocks and cudaMalloc's 256-byte
 * granularity absorb the <=15-byte overclear), so EVERY unit is a full,
 * unconditional, warp-coalesced uint4 store resolved through the chunk
 * table -- no tail predicate, no per-byte path, no branch in the loop.
 * Four units per thread per pass, interleaved at the grid stride (u,
 * u+N, u+2N, u+3N): each store instruction is coalesced and the table
 * read amortizes 4x. */
static __global__
void chunked_cuda_kernel(Index nbytes, ChunkTables t)
{
    long const nthreads = long(blockDim.x) * long(gridDim.x);
    long const units = (long(nbytes) + 15) / 16; // rounded up (padding)
    for(long pass = long(threadIdx.x) + long(blockIdx.x)*blockDim.x;
        pass < units; pass += 4*nthreads)
    {
        #pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            long const u = pass + long(j)*nthreads;
            if(u >= units)
            {
                break;
            }
            long const f = u * 16;
            Index const c = Index(f >> t.shift_data);
            *reinterpret_cast<uint4 *>(t.data[c]
                + (f - c * long(t.elems_data)))
                = make_uint4(0, 0, 0, 0);
        }
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
    (chunked_cuda_kernel)<<<blocks, threads, 0, stream>>>(
        nbytes, t);
}

} // namespace nntile::kernel::clear
