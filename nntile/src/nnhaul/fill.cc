/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/fill.cc
 * Fill operation on a StarPU buffer
 *
 * @version 1.3.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/fill.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/fill.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Fill<std::tuple<T>>::Fill():
    codelet(
        "nntile_fill",
        &Fill<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Fill<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Fill<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply fill on StarPU buffer on CPU
template<typename T>
void Fill<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto data = ::nntile::haul::buf_chunks<T>(buffers, 0);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *p) {
            kernel::fill::cpu<T>(n, args->value, p);
        },
        data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Fill<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply fill on StarPU buffer on CUDA
template<typename T>
void Fill<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto data = ::nntile::haul::buf_chunks<T>(buffers, 0);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when the buffer is chunked (or the force
    // flag is on), issue the WHOLE flat fill as
    // ONE chunk-aware kernel instead of the per-piece walk's launches.
    // The chunk base-pointer table is built on the host from the ChunkView
    // and parked in the shared device chunk-ptr ring; the kernel stages it
    // into shared memory per block. No chunk-count cap. The stored value is
    // the same scalar everywhere, so the result is cuda()'s bitwise. The
    // all-unchunked case (flag off) and a failed table upload keep the
    // historical for_each_flat_piece path verbatim.
    {
        auto buf_data = static_cast<::nnhaul::Buffer *>(buffers[0]);
        bool const all_plain =
            buf_data->chunk_count() == 1 && !chunkunits::force_chunked_kernels();
        if(!(all_plain) && args->nelems > 0)
        {
            if(buf_data->chunk_count() == 1)
            {
                // Piece C (2026-10-18c): the all-1-chunk degenerate
                // launch is the FIRST thing the chunked block does --
                // the chunk counts are read straight off the buffers,
                // before any ChunkView/RingTables work exists, so the
                // degenerate host path is as lean as the classic
                // dispatch above (gemm path-F precedent; launch
                // pointer-identical, bitwise by construction, ring
                // untouched).
                kernel::fill::cuda<T>(stream, args->nelems, args->value,
                    buf_data->get_ptr<T>());
                return;
            }
            chunkunits::ChunkView<T> const cvd(buf_data);
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
            tabs.clear();
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
            if(arena.begin(stream))
            {
                void const *ring_base = arena.upload_tables(tabs);
                if(ring_base != nullptr)
                {
                    T *const *td = arena.table_view<T *>(ring_base, o_pd);
                    kernel::fill::chunked<T>(
                        stream,
                        args->nelems,
                        args->value,
                        td,
                        Index(cvd.nchunks),
                        Index(cvd.chunk_elems));
                    arena.end(stream);
                    return;
                }
                    // (rule 11, user directive 2026-10-19) There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_fill: chunk-ptr ring cannot serve this launch (invariant "
                        "violation; the per-piece staging fallback was deleted by rule "
                        "11)\n");
                    std::abort();
            }
        }
        if(args->nelems > 0)
        {
            chunkunits::classic_dispatch_tripwire("nntile_fill", false);
        }
    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *p) {
            kernel::fill::cuda<T>(stream, n, args->value, p);
        },
        data);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Fill<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Fill<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Fill<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Fill<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Fill<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar value, ::nnhaul::Handle & data)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->value = value;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_W, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Fill<std::tuple<nntile::fp64_t>>;
template class Fill<std::tuple<nntile::fp32_t>>;
template class Fill<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Fill<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Fill<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Fill<std::tuple<nntile::bf16_t>>;
template class Fill<std::tuple<nntile::fp16_t>>;

//! Pack of fill operations for different types
fill_pack_t fill;

} // namespace nntile::haul
