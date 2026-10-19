/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelutanh_backward.cc
 * Backward approximate GeLU operation on a StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding headers
#include "nntile/nnhaul/ops/gelutanh_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/gelutanh_backward.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
GeluTanhBackward<std::tuple<T>>::GeluTanhBackward():
    codelet(
        "nntile_gelutanh_backward",
        &GeluTanhBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &GeluTanhBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &GeluTanhBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

template<typename T>
void GeluTanhBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto x = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::haul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *px, const T *pdy, T *pdx) {
            kernel::gelutanh_backward::cpu<T>(n, args->alpha, px, pdy, args->beta, pdx);
        },
        x,
        dy,
        dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluTanhBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void GeluTanhBackward<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Get arguments
    args_t const *args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto x = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::haul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::haul::buf_chunks<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when at least one operand is chunked (or
    // the force flag is on), issue the WHOLE flat
    // operation as ONE chunk-aware kernel instead of the per-piece walk's
    // launches. The chunk base-pointer tables are built on the host from
    // the ChunkViews and parked in the shared device chunk-ptr ring; the
    // kernel reads the table per-thread by shift/mask (rule 10: no smem staging). No chunk-count cap.
    // The per-element expression is identical (including the beta special
    // cases), so the result is cuda()'s bitwise. The all-unchunked case
    // (flag off) and a failed table upload keep the historical
    // for_each_flat_piece path verbatim.
    {
        auto buf_x = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dy = static_cast<::nnhaul::Buffer *>(buffers[1]);
        auto buf_dx = static_cast<::nnhaul::Buffer *>(buffers[2]);
        bool const all_plain = (buf_x->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
            && (buf_dy->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
            && (buf_dx->chunk_count() == 1 && !chunkunits::force_chunked_kernels());
        if(!(all_plain) && args->nelems > 0)
        {
            if(buf_x->chunk_count() == 1 && buf_dy->chunk_count() == 1 && buf_dx->chunk_count() == 1)
            {
                // Piece C (2026-10-18c): the all-1-chunk degenerate
                // launch is the FIRST thing the chunked block does --
                // the chunk counts are read straight off the buffers,
                // before any ChunkView/RingTables work exists, so the
                // degenerate host path is as lean as the classic
                // dispatch above (gemm path-F precedent; launch
                // pointer-identical, bitwise by construction, ring
                // untouched).
                kernel::gelutanh_backward::cuda<T>(stream, args->nelems,
                    args->alpha, buf_x->get_ptr<T>(), buf_dy->get_ptr<T>(), args->beta,
                    buf_dx->get_ptr<T>());
                return;
            }
            chunkunits::ChunkView<T> const cvx(buf_x);
            chunkunits::ChunkView<T> const cvdy(buf_dy);
            chunkunits::ChunkView<T> const cvdx(buf_dx);
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
            tabs.clear();
            std::size_t const o_px = tabs.add(cvx);
            std::size_t const o_pdy = tabs.add(cvdy);
            std::size_t const o_pdx = tabs.add(cvdx);
            chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
            if(arena.begin(stream))
            {
                void const *ring_base = arena.upload_tables(tabs);
                if(ring_base != nullptr)
                {
                    const T *const *tx = arena.table_view<const T *>(
                        ring_base, o_px);
                    const T *const *tdy = arena.table_view<const T *>(
                        ring_base, o_pdy);
                    T *const *tdx = arena.table_view<T *>(ring_base, o_pdx);
                    kernel::gelutanh_backward::chunked<T>(
                        stream,
                        args->nelems,
                        args->alpha,
                        tx,
                        Index(cvx.nchunks),
                        tdy,
                        Index(cvdy.nchunks),
                        args->beta,
                        tdx,
                        Index(cvdx.nchunks),
                        Index(cvx.chunk_elems),
                        Index(cvdy.chunk_elems),
                        Index(cvdx.chunk_elems));
                    arena.end(stream);
                    return;
                }
                    // (rule 11, user directive 2026-10-19) There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_gelutanh_backward: chunk-ptr ring cannot serve this launch (invariant "
                        "violation; the per-piece staging fallback was deleted by rule "
                        "11)\n");
                    std::abort();
            }
        }
        if(args->nelems > 0)
        {
            chunkunits::classic_dispatch_tripwire("nntile_gelutanh_backward",
                false);
        }
    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *px, const T *pdy, T *pdx) {
            kernel::gelutanh_backward::cuda<T>(
                stream,
                n,
                args->alpha,
                px,
                pdy,
                args->beta,
                pdx
            );
        },
        x,
        dy,
        dx);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void GeluTanhBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanhBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanhBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t GeluTanhBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void GeluTanhBackward<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha,
        ::nnhaul::Handle & x, ::nnhaul::Handle & dy, Scalar beta, ::nnhaul::Handle & dx)
//! Insert gelutanh_backward task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    starpu_data_access_mode dx_mode;
    if(beta == 0.0)
    {
        dx_mode = STARPU_W;
    }
    else
    {
        dx_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &x }, { STARPU_R, &dy }, { dx_mode, &dx } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluTanhBackward<std::tuple<nntile::fp64_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluTanhBackward<std::tuple<nntile::bf16_t>>;
template class GeluTanhBackward<std::tuple<nntile::fp16_t>>;

//! Pack of gelutanh_backward operations for different types
gelutanh_backward_pack_t gelutanh_backward;

} // namespace nntile::haul
