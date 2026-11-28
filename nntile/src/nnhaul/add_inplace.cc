/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/add_inplace.cc
 * Add operation on a NNHaul buffers
 *
 * @version 2.0.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"

//! NNHaul wrappers for add_inplace operation
namespace nntile::nnhaul
{

//! Constructor
template<typename T>
AddInplace<std::tuple<T>>::AddInplace():
    codelet(
        "nntile_add_inplace",
        &AddInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AddInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AddInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add operation for NNHaul buffers in CPU
template<typename T>
void AddInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
            kernel::add_inplace::cpu<T>(n, args->alpha, s, args->beta, d);
        },
        src,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add for NNHaul buffers on CUDA
template<typename T>
void AddInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_add_inplace");
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when at least one operand is chunked, issue the WHOLE flat addition as ONE
    // chunk-aware kernel instead of the per-piece walk's launches. The chunk
    // base-pointer tables are built on the host and parked in the shared
    // device chunk-ptr table (no chunk-count cap); the
    // kernel reads the table per-thread by shift/mask . Elements are
    // independent and the expression is identical, so the result is cuda()'s
    // bitwise. The all-unchunked case (and a zero-element launch) keeps the
    // historical for_each_flat_piece path; any table failure aborts.
    {
        auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
        bool const all_plain = (buf_src->chunk_count() == 1)
            && (buf_dst->chunk_count() == 1);
        if(!(all_plain) && args->nelems > 0)
        {
            chunkunits::ChunkView<T> const cvs(buf_src);
            chunkunits::ChunkView<T> const cvd(buf_dst);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_ps = tabs.add(cvs);
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrTable &arena =
                chunkunits::chunk_ptr_table();
            if(arena.begin(stream))
            {
                void const *table_base = arena.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    const T *const *ts = arena.table_view<const T *>(
                        table_base, o_ps);
                    T *const *td = arena.table_view<T *>(table_base, o_pd);
                    kernel::add_inplace::chunked<T>(
                        stream,
                        args->nelems,
                        args->alpha,
                        ts,
                        Index(cvs.nchunks),
                        args->beta,
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
                        "nntile_add_inplace: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::add_inplace::cuda<T>(stream, n, args->alpha, s, args->beta, d);
        },
        src,
        dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AddInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t AddInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit add_inplace task
template<typename T>
void AddInplace<std::tuple<T>>::submit(int worker_hint,
    Index nelems,
    Scalar alpha,
    ::nnhaul::Handle & src,
    Scalar beta,
    ::nnhaul::Handle & dst
)
{
    // If alpha is zero then reduce to scale_inplace
    if(alpha == 0.0)
    {
        scale_inplace.submit<std::tuple<T>>(worker_hint, nelems, beta, dst);
        return;
    }
    // If beta is zero this function reduces to scale
    if(beta == 0.0)
    {
        scale.submit<std::tuple<T>>(worker_hint, nelems, alpha, src, dst);
        return;
    }
    // Access mode for the dst handle
    enum data_access_mode dst_mode;
    if(beta == 1.0)
    {
        dst_mode = static_cast<data_access_mode>(ACCESS_RW | ACCESS_COMMUTE);
    }
    else
    {
        dst_mode = ACCESS_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = 2 * nelems;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddInplace<std::tuple<nntile::fp64_t>>;
template class AddInplace<std::tuple<nntile::fp32_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddInplace<std::tuple<nntile::bf16_t>>;
template class AddInplace<std::tuple<nntile::fp16_t>>;

//! Pack of add_inplace operations for different types
add_inplace_pack_t add_inplace;

} // namespace nntile::nnhaul
