/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/add.cc
 * Add operation on a NNHaul buffers
 *
 * @version 2.0.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/scale.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
Add<std::tuple<T>>::Add():
    codelet(
        "nntile_add",
        &Add<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Add<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Add<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add operation for NNHaul buffers in CPU
template<typename T>
void Add<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src1 = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto src2 = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s1, const T *s2, T *d) {
            kernel::add::cpu<T>(n, args->alpha, s1, args->beta, s2, d);
        },
        src1,
        src2,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add for NNHaul buffers on CUDA
template<typename T>
void Add<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_add");
    auto src1 = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto src2 = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Chunked single-launch path: when at least one operand is chunked, issue the WHOLE flat addition as ONE
    // chunk-aware kernel instead of the per-piece walk's launches. The chunk
    // base-pointer tables are built on the host from the ChunkViews and
    // parked in the shared device chunk-ptr table; the
    // kernel reads the table per-thread by shift/mask . No chunk-count cap.
    // Elements are independent and the expression is identical, so the
    // result is cuda()'s bitwise. The all-unchunked case (and a zero-element
    // launch) keeps the historical for_each_flat_piece path; any table
    // failure aborts.
    {
        auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
        auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
        bool const all_plain = (buf_src1->chunk_count() == 1)
            && (buf_src2->chunk_count() == 1) && (buf_dst->chunk_count() == 1);
        if(!(all_plain) && args->nelems > 0)
        {
            chunkunits::ChunkView<T> const cv1(buf_src1);
            chunkunits::ChunkView<T> const cv2(buf_src2);
            chunkunits::ChunkView<T> const cvd(buf_dst);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_p1 = tabs.add(cv1);
            std::size_t const o_p2 = tabs.add(cv2);
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrTable &arena =
                chunkunits::chunk_ptr_table();
            if(arena.begin(stream))
            {
                void const *table_base = arena.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    const T *const *t1 = arena.table_view<const T *>(
                        table_base, o_p1);
                    const T *const *t2 = arena.table_view<const T *>(
                        table_base, o_p2);
                    T *const *td = arena.table_view<T *>(table_base, o_pd);
                    kernel::add::chunked<T>(
                        stream,
                        args->nelems,
                        args->alpha,
                        t1,
                        Index(cv1.nchunks),
                        args->beta,
                        t2,
                        Index(cv2.nchunks),
                        td,
                        Index(cvd.nchunks),
                        Index(cv1.chunk_elems),
                        Index(cv2.chunk_elems),
                        Index(cvd.chunk_elems));
                    arena.end(stream);
                    return;
                }
                    // There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_add: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s1, const T *s2, T *d) {
            kernel::add::cuda<T>(
                stream,
                n,
                args->alpha,
                s1,
                args->beta,
                s2,
                d);
        },
        src1,
        src2,
        dst);
}

template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t Add<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Add<std::tuple<T>>::submit(int worker_hint,
    Index nelems,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    Scalar beta,
    ::nnhaul::Handle & src2,
    ::nnhaul::Handle & dst
)
{
    constexpr Scalar zero = 0;
    // If beta is zero this function reduces to scale
    if(beta == zero)
    {
        // dst = alpha*src1
        scale.submit<std::tuple<T>>(worker_hint, nelems, alpha, src1, dst);
        return;
    }
    // If beta is non-zero and alpha is zero then reduce to scale
    if(alpha == zero)
    {
        // dst = beta*src2
        scale.submit<std::tuple<T>>(worker_hint, nelems, beta, src2, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * 3 * nelems;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src1 }, { ACCESS_R, &src2 }, { ACCESS_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Add<std::tuple<nntile::fp64_t>>;
template class Add<std::tuple<nntile::fp32_t>>;
template class Add<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Add<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Add<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Add<std::tuple<nntile::bf16_t>>;
template class Add<std::tuple<nntile::fp16_t>>;

//! Pack of add operations for different types
add_pack_t add;

} // namespace nntile::nnhaul
