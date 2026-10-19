/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/copy/cuda.cu
 * Copy operation on buffers on CUDA
 *
 * @version 1.2.0
 * */

#include "nntile/kernel/copy/cuda.hh"
#include "nntile/kernel/cuda.hh"

#include <cstdint>

namespace nntile::kernel::copy
{

//! Chunk descriptors for the one-launch chunked copy (see cuda.hh).
struct ChunkTables
{
    const unsigned char *const* src; //!< device-ring view of src pointers
    Index nsrc;
    unsigned char *const* dst; //!< device-ring view of dst pointers
    Index ndst;
    Index elems_src; //!< bytes in a full src chunk
    Index elems_dst; //!< bytes in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat byte offset (power-of-two shift when possible).
static __device__ __forceinline__ long cpy_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware copy: semantically a memcpy of the whole flat byte range,
//! issued as ONE launch. Each thread owns one 16-byte unit of the flat
//! range (the tail unit may be shorter): a unit that lies inside one
//! chunk of BOTH operands at 16-byte-aligned addresses moves as one uint4
//! load/store; any other unit (straddling a chunk boundary or misaligned
//! in either operand) resolves EVERY byte through both chunk tables (a
//! straddling unit's bytes belong to two different allocations per
//! operand). Copying has no value freedom, so the result is trivially
//! bitwise the classic cudaMemcpyAsync's.
static __global__
void chunked_cuda_kernel(Index nbytes, ChunkTables t)
{
    long const u = long(threadIdx.x) + long(blockIdx.x)*blockDim.x;
    long const f = u * 16;
    if(f >= long(nbytes))
    {
        return;
    }
    Index const cs = Index(cpy_chunk_index(f, t.elems_src, t.shift_src));
    Index const cd = Index(cpy_chunk_index(f, t.elems_dst, t.shift_dst));
    long const offs = f - cs * long(t.elems_src);
    long const offd = f - cd * long(t.elems_dst);
    const unsigned char *s = t.src[cs] + offs;
    unsigned char *d = t.dst[cd] + offd;
    long const n = long(nbytes) - f < 16 ? long(nbytes) - f : 16;
    if(n == 16 && offs + 16 <= long(t.elems_src)
        && offd + 16 <= long(t.elems_dst)
        && (reinterpret_cast<std::uintptr_t>(s) & 15) == 0
        && (reinterpret_cast<std::uintptr_t>(d) & 15) == 0)
    {
        *reinterpret_cast<uint4 *>(d) = *reinterpret_cast<const uint4 *>(s);
    }
    else
    {
        for(long b = 0; b < n; ++b)
        {
            long const fb = f + b;
            Index const csb = Index(cpy_chunk_index(fb, t.elems_src,
                t.shift_src));
            Index const cdb = Index(cpy_chunk_index(fb, t.elems_dst,
                t.shift_dst));
            t.dst[cdb][fb - cdb * long(t.elems_dst)] =
                t.src[csb][fb - csb * long(t.elems_src)];
        }
    }
}

void chunked(cudaStream_t stream, Index nbytes,
        const unsigned char* const* src_chunks, Index nsrc,
        unsigned char* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst)
    noexcept
{
    if(nbytes <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<unsigned char>(t.nsrc, elems_src, "nntile_copy");
    t.shift_dst = chunk_shift_of<unsigned char>(t.ndst, elems_dst, "nntile_copy");
    Index const units = (nbytes + 15) / 16;
    dim3 blocks((units+255)/256), threads(256);
    (chunked_cuda_kernel)<<<blocks, threads,
        0, stream>>>(nbytes, t);
}

} // namespace nntile::kernel::copy
