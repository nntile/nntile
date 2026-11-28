/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/clear.cc
 * Clear a NNHaul buffer
 *
 * @version 2.0.0
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

namespace nntile::nnhaul
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
    constexpr std::array<data_access_mode, 1> modes = {
        ACCESS_W
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Clear a NNHaul buffer on CPU
void Clear::cpu(void *buffers[], void *cl_args)
    noexcept
{
    (void)cl_args;
    // Get interfaces
    auto data = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 0);
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
    auto data = ::nntile::nnhaul::buf_chunks<unsigned char>(buffers, 0);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // ONE launch for the
    // chunked clear: per-chunk cudaMemsetAsync does not scale (one launch
    // per chunk; launch overhead would dominate on a fast card). The
    // kernel rounds the clear up
    // to a 16-byte multiple (allocation padding absorbs the <=15 B) so
    // every unit is one unconditional uint4 store through the chunk
    // table. Degenerate 1-chunk case: the single memset.
    {
        auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
        bool const all_plain =
            buf_data->chunk_count() == 1;
        std::size_t const nbytes = buf_data->nbytes();
        if(!(all_plain) && nbytes > 0)
        {
            chunkunits::ChunkView<unsigned char> const cvd(buf_data);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrTable &table = chunkunits::chunk_ptr_table();
            if(table.begin(stream))
            {
                void const *table_base = table.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    unsigned char *const *td =
                        table.table_view<unsigned char *>(table_base, o_pd);
                    kernel::clear::chunked(stream, Index(nbytes), td,
                        Index(cvd.nchunks), Index(cvd.chunk_elems));
                    table.end(stream);
                    return;
                }
                //  No per-piece fallback exists: a launch the
                // table cannot serve is an invariant violation.
                std::fprintf(stderr,
                    "nntile_clear: chunk-ptr table cannot serve this "
                    "launch \n");
                std::abort();
            }
            std::fprintf(stderr,
                "nntile_clear: chunk-ptr table init failed \n");
            std::abort();
        }

    }
    for (std::size_t i = 0; i < data.count(); ++i)
    {
        cudaMemsetAsync(data.ptr(i), 0, data.elems(i), stream);
    }
}
#endif // NNTILE_USE_CUDA

//! Submit clear task
void Clear::submit(int worker_hint, ::nnhaul::Handle & data)
{
    // Submit task
    std::size_t nnhaul_nbytes = data.nbytes();
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_W, &data } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}

//! Clear operation object
Clear clear;

} // namespace nntile::nnhaul
