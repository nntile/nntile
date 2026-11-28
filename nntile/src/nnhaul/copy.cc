/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/copy.cc
 * Copy NNHaul buffers
 *
 * @version 2.0.0
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

namespace nntile::nnhaul
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

//! Copy NNHaul buffers on CPU
void Copy::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 1);
    // Copy per flat chunk piece (single piece if unchunked); src and dst
    // may chunk differently (same numel, different dtype)
    ::nntile::nnhaul::for_each_flat_piece(
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
    auto src = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 1);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    {
        auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
        bool const all_plain = (buf_src->chunk_count() == 1)
            && (buf_dst->chunk_count() == 1);
        if(!(all_plain) && args->nbytes > 0)
        {
            // Chunked single-launch path: the whole multi-chunk range is
            // issued as ONE chunk-aware kernel reading the shared device
            // chunk-pointer table; the per-element math matches cuda().
            chunkunits::ChunkView<unsigned char> const cvs(buf_src);
            chunkunits::ChunkView<unsigned char> const cvd(buf_dst);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_ps = tabs.add(cvs);
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
            if(arena.begin(stream))
            {
                void const *table_base = arena.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    const unsigned char *const *ts = arena.table_view<const unsigned char *>(
                        table_base, o_ps);
                    unsigned char *const *td = arena.table_view<unsigned char *>(
                        table_base, o_pd);
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
                    // There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_copy: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    ::nntile::nnhaul::for_each_flat_piece(
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

void Copy::submit(int worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert copy task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Get arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nbytes = src.nbytes();
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { ACCESS_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

//! Copy operation object
Copy copy;

} // namespace nntile::nnhaul
