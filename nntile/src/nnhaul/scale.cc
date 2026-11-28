/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/scale.cc
 * Scale operation on a NNHaul buffers
 *
 * @version 2.0.0
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

namespace nntile::nnhaul
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

//! NNHaul wrapper for kernel::scale::cpu<T>
template<typename T>
void Scale<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
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
//! NNHaul wrapper for kernel::scale::cuda<T>
template<typename T>
void Scale<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_scale");
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    {
        auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
        bool const all_plain = (buf_src->chunk_count() == 1)
            && (buf_dst->chunk_count() == 1);
        if(!(all_plain) && args->nelems > 0)
        {
            // Chunked single-launch path: the whole multi-chunk range is
            // issued as ONE chunk-aware kernel reading the shared device
            // chunk-pointer table; the per-element math matches cuda().
            chunkunits::ChunkView<T> const cvs(buf_src);
            chunkunits::ChunkView<T> const cvd(buf_dst);
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
                    const T *const *ts = arena.table_view<const T *>(
                        table_base, o_ps);
                    T *const *td = arena.table_view<T *>(table_base, o_pd);
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
                    // There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_scale: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
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
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Scale<std::tuple<T>>::submit(int worker_hint,
    Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero, function reduces to clear
    if(alpha == zero)
    {
        clear.submit(worker_hint, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { ACCESS_W, &dst } }, args, sizeof(*args));
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

} // namespace nntile::nnhaul
