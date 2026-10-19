/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/clear.hh"

// Standard libraries
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Other NNTile headers
#ifdef NNTILE_USE_CUDA
#include "nntile/kernel/clear/cuda.hh"
#endif // NNTILE_USE_CUDA
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

//! Constructor
Clear::Clear():
    codelet(
        "nntile_clear",
        &Clear::cpu,
#ifdef NNTILE_USE_CUDA
        &Clear::cuda,
#else
        nullptr,
#endif
        nullptr)
{
    // Modes cannot be variable for clear operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 1> modes = {
        STARPU_W
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Clear a StarPU buffer on CPU
void Clear::cpu(void *buffers[], void *cl_args)
    noexcept
{
    (void)cl_args;
    // Get interfaces
    auto data = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    // Clear buffer chunk by chunk (single chunk if unchunked)
    for (std::size_t i = 0; i < data.count(); ++i)
    {
        std::memset(data.ptr(i), 0, data.elems(i));
    }
}

#ifdef NNTILE_USE_CUDA
//! Clear an NNHaul buffer on CUDA
void Clear::cuda(void *buffers[], void *cl_args)
    noexcept
{
    (void)cl_args;
    auto data = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // (user directive, 2026-10-19, second form) ONE launch for the
    // chunked clear: per-chunk cudaMemsetAsync does not scale (a 16 MiB
    // chunk is ~4 us of fill at 3.6 TB/s -- launch overhead would
    // dominate on H200-class bandwidth). The kernel rounds the clear up
    // to a 16-byte multiple (allocation padding absorbs the <=15 B) so
    // every unit is one unconditional uint4 store through the chunk
    // table. Degenerate 1-chunk case: the single memset.
    {
        auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
        bool const all_plain =
            buf_data->chunk_count() == 1
            && !chunkunits::force_chunked_kernels();
        std::size_t const nbytes = buf_data->nbytes();
        if(!(all_plain) && nbytes > 0)
        {
            if(buf_data->chunk_count() == 1)
            {
                ::cudaMemsetAsync(buf_data->get_ptr<unsigned char>(), 0,
                    nbytes, stream);
                return;
            }
            chunkunits::ChunkView<unsigned char> const cvd(buf_data);
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
            tabs.clear();
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrRing &ring = chunkunits::chunk_ptr_ring();
            if(ring.begin(stream))
            {
                void const *ring_base = ring.upload_tables(tabs);
                if(ring_base != nullptr)
                {
                    unsigned char *const *td =
                        ring.table_view<unsigned char *>(ring_base, o_pd);
                    kernel::clear::chunked(stream, Index(nbytes), td,
                        Index(cvd.nchunks), Index(cvd.chunk_elems));
                    ring.end(stream);
                    return;
                }
                // (rule 11) No per-piece fallback exists: a launch the
                // ring cannot serve is an invariant violation.
                std::fprintf(stderr,
                    "nntile_clear: chunk-ptr ring cannot serve this "
                    "launch (invariant violation; rule 11)\n");
                std::abort();
            }
            std::fprintf(stderr,
                "nntile_clear: chunk-ptr ring init failed (rule 11)\n");
            std::abort();
        }
        if(nbytes > 0)
        {
            chunkunits::classic_dispatch_tripwire("nntile_clear", false);
        }
    }
    for (std::size_t i = 0; i < data.count(); ++i)
    {
        cudaMemsetAsync(data.ptr(i), 0, data.elems(i), stream);
    }
}
#endif // NNTILE_USE_CUDA

//! Submit clear task
void Clear::submit(int starpu_worker_hint, ::nnhaul::Handle & data)
{
    // Submit task
    std::size_t nnhaul_nbytes = data.nbytes();
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_W, &data } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}

//! Clear operation object
Clear clear;

} // namespace nntile::haul
