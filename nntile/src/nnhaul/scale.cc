/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale.cc
 * Scale operation on a StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Scale<std::tuple<T>>::Scale():
    codelet(
        "nntile_scale",
        &Scale<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Scale<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Scale<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::scale::cpu<T>
template<typename T>
void Scale<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<T>(buffers, 1);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::scale::cpu<T>(n, args->alpha, s, d);
        },
        src,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Scale<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::scale::cuda<T>
template<typename T>
void Scale<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when at least one operand is chunked (or
    // the force flag is on), issue the WHOLE flat
    // operation as ONE chunk-aware kernel instead of the per-piece walk's
    // launches. The chunk base-pointer tables are built on the host from
    // the ChunkViews and parked in the shared device chunk-ptr ring; the
    // kernel reads the table per-thread by shift/mask (rule 10: no smem staging). No chunk-count cap.
    // The per-element expression is identical, so the result is cuda()'s
    // bitwise. The all-unchunked case (flag off) and a failed table upload
    // keep the historical for_each_flat_piece path verbatim.
    {
        auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
        bool const all_plain = (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
            && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels());
        if(!(all_plain) && args->nelems > 0)
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
                kernel::scale::cuda<T>(stream, args->nelems, args->alpha,
                    buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
                return;
            }
            chunkunits::ChunkView<T> const cvs(buf_src);
            chunkunits::ChunkView<T> const cvd(buf_dst);
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
                    const T *const *ts = arena.table_view<const T *>(
                        ring_base, o_ps);
                    T *const *td = arena.table_view<T *>(ring_base, o_pd);
                    kernel::scale::chunked<T>(
                        stream,
                        args->nelems,
                        args->alpha,
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
                        "nntile_scale: chunk-ptr ring cannot serve this launch (invariant "
                        "violation; the per-piece staging fallback was deleted by rule "
                        "11)\n");
                    std::abort();
            }
        }
        if(args->nelems > 0)
        {
            chunkunits::classic_dispatch_tripwire("nntile_scale", false);
        }
    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::scale::cuda<T>(stream, n, args->alpha, s, d);
        },
        src,
        dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Scale<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scale<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scale<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scale tasks that depends only on cl_arg
template<typename T>
std::uint64_t Scale<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Scale<std::tuple<T>>::submit(int starpu_worker_hint,
    Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero, function reduces to clear
    if(alpha == zero)
    {
        clear.submit(starpu_worker_hint, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Scale<std::tuple<nntile::fp64_t>>;
template class Scale<std::tuple<nntile::fp32_t>>;
template class Scale<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Scale<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Scale<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Scale<std::tuple<nntile::bf16_t>>;
template class Scale<std::tuple<nntile::fp16_t>>;

//! Pack of scale operations for different types
scale_pack_t scale;

} // namespace nntile::haul
