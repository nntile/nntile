/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/gelutanh_backward.cc
 * Backward approximate GeLU operation on a NNHaul buffer
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

namespace nntile::nnhaul
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
    auto x = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
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
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems),
        "nntile_gelutanh_backward");
    // Get interfaces
    auto x = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    {
        auto buf_x = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dy = static_cast<::nnhaul::Buffer *>(buffers[1]);
        auto buf_dx = static_cast<::nnhaul::Buffer *>(buffers[2]);
        bool const all_plain = (buf_x->chunk_count() == 1)
            && (buf_dy->chunk_count() == 1)
            && (buf_dx->chunk_count() == 1);
        if(!(all_plain) && args->nelems > 0)
        {
            // Chunked single-launch path: the whole multi-chunk range is
            // issued as ONE chunk-aware kernel reading the shared device
            // chunk-pointer table; the per-element math matches cuda().
            chunkunits::ChunkView<T> const cvx(buf_x);
            chunkunits::ChunkView<T> const cvdy(buf_dy);
            chunkunits::ChunkView<T> const cvdx(buf_dx);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_px = tabs.add(cvx);
            std::size_t const o_pdy = tabs.add(cvdy);
            std::size_t const o_pdx = tabs.add(cvdx);
            chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
            if(arena.begin(stream))
            {
                void const *table_base = arena.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    const T *const *tx = arena.table_view<const T *>(
                        table_base, o_px);
                    const T *const *tdy = arena.table_view<const T *>(
                        table_base, o_pdy);
                    T *const *tdx = arena.table_view<T *>(table_base, o_pdx);
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
                    // There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_gelutanh_backward: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
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
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void GeluTanhBackward<std::tuple<T>>::submit(int worker_hint, Index nelems, Scalar alpha,
        ::nnhaul::Handle & x, ::nnhaul::Handle & dy, Scalar beta, ::nnhaul::Handle & dx)
//! Insert gelutanh_backward task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    data_access_mode dx_mode;
    if(beta == 0.0)
    {
        dx_mode = ACCESS_W;
    }
    else
    {
        dx_mode = ACCESS_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &x }, { ACCESS_R, &dy }, { dx_mode, &dx } }, args, sizeof(*args));
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

} // namespace nntile::nnhaul
