/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/copy.cc
 * Copy StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/copy.hh"

// Standard libraries
#include <cstring>

// Other NNTile headers
#ifdef NNTILE_USE_CUDA
#include "nntile/kernel/copy/cuda.hh"
#endif // NNTILE_USE_CUDA
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

//! Constructor
Copy::Copy():
    codelet(
        "nntile_copy",
        &Copy::cpu,
#ifdef NNTILE_USE_CUDA
        &Copy::cuda,
#else
        nullptr,
#endif
        nullptr)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Copy StarPU buffers on CPU
void Copy::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<unsigned char>(buffers, 1);
    // Copy per flat chunk piece (single piece if unchunked); src and dst
    // may chunk differently (same numel, different dtype)
    ::nntile::haul::for_each_flat_piece(
        args->nbytes,
        [&](std::size_t n, const unsigned char *s, unsigned char *d) {
            std::memcpy(d, s, n);
        },
        src,
        dst);
}

#ifdef NNTILE_USE_CUDA
//! Copy NNHaul buffers on CUDA
void Copy::cuda(void *buffers[], void *cl_args)
    noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    auto src = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<unsigned char>(buffers, 1);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when at least one operand is chunked (or
    // the force flag is on), move the WHOLE flat
    // byte range as ONE chunk-aware kernel instead of the per-piece walk's
    // memcpys. The merged chunk table rides the device ring (ONE
    // event-ordered control-stream copy per launch; rule 10: no smem
    // staging) and the kernel reads it per-thread by shift/mask. No
    // chunk-count cap. Copying has no value freedom, so the result is
    // trivially the memcpys' bitwise. The all-unchunked case (flag off)
    // keeps the historical for_each_flat_piece path; under the flag a
    // failed upload ABORTS (rule 11 -- no fallback).
    {
        auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
        bool const all_plain = (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
            && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels());
        if(!(all_plain) && args->nbytes > 0)
        {
            if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
            {
                // Piece C (2026-10-18c): the all-1-chunk degenerate
                // launch is the FIRST thing the chunked block does --
                // the chunk counts are read straight off the buffers,
                // before any ChunkView/RingTables work exists, so the
                // degenerate host path is as lean as the classic
                // dispatch above (gemm path-F precedent; launch
                // pointer-identical, bitwise by construction, ring
                // untouched).
                ::cudaMemcpyAsync(buf_dst->get_ptr<unsigned char>(), buf_src->get_ptr<unsigned char>(), args->nbytes,
                    ::cudaMemcpyDeviceToDevice, stream);
                return;
            }
            chunkunits::ChunkView<unsigned char> const cvs(buf_src);
            chunkunits::ChunkView<unsigned char> const cvd(buf_dst);
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
            tabs.clear();
            std::size_t const o_ps = tabs.add(cvs);
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
            if(arena.begin(stream))
            {
                void const *ring_base = arena.upload_tables(tabs);
                if(ring_base != nullptr)
                {
                    const unsigned char *const *ts = arena.table_view<const unsigned char *>(
                        ring_base, o_ps);
                    unsigned char *const *td = arena.table_view<unsigned char *>(
                        ring_base, o_pd);
                    kernel::copy::chunked(
                        stream,
                        Index(args->nbytes),
                        ts,
                        Index(cvs.nchunks),
                        td,
                        Index(cvd.nchunks),
                        Index(cvs.chunk_elems),
                        Index(cvd.chunk_elems));
                    arena.end(stream);
                    return;
                }
                    // (rule 11, user directive 2026-10-19) There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_copy: chunk-ptr ring cannot serve this launch (invariant "
                        "violation; the per-piece staging fallback was deleted by rule "
                        "11)\n");
                    std::abort();
            }
        }
        if(args->nbytes > 0)
        {
            chunkunits::classic_dispatch_tripwire("nntile_copy",
                false);
        }
    }
    ::nntile::haul::for_each_flat_piece(
        args->nbytes,
        [&](std::size_t n, const unsigned char *s, unsigned char *d) {
            cudaMemcpyAsync(
                d,
                s,
                n,
                cudaMemcpyDeviceToDevice,
                stream);
        },
        src,
        dst);
}
#endif // NNTILE_USE_CUDA

void Copy::submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert copy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Get arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nbytes = src.nbytes();
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

//! Copy operation object
Copy copy;

} // namespace nntile::haul
