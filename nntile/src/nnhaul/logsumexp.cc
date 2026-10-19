/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/logsumexp.cc
 * Log of sum of exponents for StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/logsumexp.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/logsumexp.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked logsumexp driver (chunk_units.hh scheme, cf. gemm.cc)
//
// Kernel index map (derived from kernel::logsumexp::cpu/cuda, NOT
// assumed): lse[i] = mse[2*i] + log(mse[2*i+1]) - an ELEMENTWISE 2-to-1
// map, so pieces are runs of units i whose mse read [2*i, 2*i+2) lies in
// one mse chunk and whose lse write [i, i+1) lies in one lse chunk (the
// walk of rows-of-2 over mse against rows-of-1 over lse). A maxsumexp
// PAIR straddling an mse chunk boundary is gathered into a bounded
// 2-element temp. Bitwise equal to the plain call (per-element
// operations unchanged). Zero new kernels; all-unchunked buffers keep
// the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void logsumexp_chunked(Index nelems, ::nnhaul::Buffer *buf_mse,
    ::nnhaul::Buffer *buf_lse, Arena &arena,
    Sub &&sub)
{
    chunkunits::ChunkView<T> const cv_mse(buf_mse);
    chunkunits::ChunkView<T> const cv_lse(buf_lse);
    chunkunits::UnitSpan<T> const span_mse{&cv_mse, std::size_t{2}, Index{1}};
    chunkunits::UnitSpan<T> const span_lse{&cv_lse, std::size_t{1}, Index{1}};
    bool const chunked_mse = cv_mse.nchunks > 1;
    T *temp_mse = chunked_mse ? arena.alloc(std::size_t{2}) : nullptr;
    chunkunits::walk_unit_runs(Index{1}, nelems,
        [&](Index i, Index len, bool /*first*/, bool mse_str,
            bool /*lse_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            T const *pm = nullptr;
            if(mse_str)
            {
                chunkunits::gather_range(
                    cv_mse, std::size_t{2} * i, std::size_t{2}, temp_mse,
                    arena);
                pm = temp_mse;
            }
            else
            {
                pm = span_mse.ptr(i);
            }
            T *pl = span_lse.ptr(i);
            if(pm == nullptr || pl == nullptr || !arena.ok_state())
            {
                return;
            }
            sub(len, pm, pl);
        },
        span_mse, span_lse);
}

} // namespace

//! Constructor
template<typename T>
LogSumExp<std::tuple<T>>::LogSumExp():
    codelet(
        "nntile_logsumexp",
        &LogSumExp<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &LogSumExp<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &LogSumExp<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply logsumexp operation for NNHaul buffers in CPU
template<typename T>
void LogSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_lse = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if((buf_mse->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_lse->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::logsumexp::cpu<T>(
            args->nelems, buf_mse->get_ptr<T>(), buf_lse->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        logsumexp_chunked<T>(
            args->nelems,
            buf_mse,
            buf_lse,
            arena,
            [](Index nn, T const *pm, T *pl) {
                kernel::logsumexp::cpu<T>(nn, pm, pl);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_logsumexp: cpu chunked driver aborted after an "
            "allocation failure; logsumexp left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void LogSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
template<typename T>
void LogSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_lse = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_mse->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_lse->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag (the
        // tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_logsumexp", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::logsumexp::cuda<T>(
            stream, args->nelems, buf_mse->get_ptr<T>(), buf_lse->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: issue the WHOLE 2-to-1 reduction as ONE
    // chunk-aware kernel (each element reads its maxsumexp pair through the
    // mse chunk table -- the pair may straddle chunks -- and writes its
    // logsumexp through the lse table). The tables are parked in the shared
    // device chunk-ptr ring; no chunk-count cap. The per-element expression
    // is identical, so the result is cuda()'s bitwise. The piece driver
    // below stays as the ring-overflow fallback and for degenerate shapes.
    if(args->nelems > 0)
    {
        // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
        // the FIRST thing the chunked block does -- the chunk counts
        // are read straight off the buffers, before any
        // ChunkView/RingTables work exists, so the degenerate host
        // path is as lean as the classic dispatch above (gemm path-F
        // precedent; launch pointer-identical, bitwise by
        // construction, ring untouched).
        if(buf_mse->chunk_count() == 1 && buf_lse->chunk_count() == 1)
        {
            kernel::logsumexp::cuda<T>(
                stream, args->nelems, buf_mse->get_ptr<T>(),
                buf_lse->get_ptr<T>());
            return;
        }
        chunkunits::ChunkView<T> const cv_mse(buf_mse);
        chunkunits::ChunkView<T> const cv_lse(buf_lse);
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_pm = tabs.add(cv_mse);
        std::size_t const o_pl = tabs.add(cv_lse);
        chunkunits::ChunkPtrRing &ring = chunkunits::chunk_ptr_ring();
        if(ring.begin(stream))
        {
            void const *ring_base = ring.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *tm = ring.table_view<const T *>(
                    ring_base, o_pm);
                T *const *tl = ring.table_view<T *>(ring_base, o_pl);
                kernel::logsumexp::chunked<T>(
                    stream,
                    args->nelems,
                    tm,
                    Index(cv_mse.nchunks),
                    tl,
                    Index(cv_lse.nchunks),
                    Index(cv_mse.chunk_elems),
                    Index(cv_lse.chunk_elems));
                ring.end(stream);
                return;
            }
        }
    }
        // (rule 11, user directive 2026-10-19) There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_logsumexp: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void LogSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LogSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t LogSumExp<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit logsumexp task
template<typename T>
void LogSumExp<std::tuple<T>>::submit(int starpu_worker_hint,
        Index nelems, ::nnhaul::Handle & maxsumexp, ::nnhaul::Handle & logsumexp)
//! Insert logsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; logsumexp never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &maxsumexp }, { STARPU_W, &logsumexp } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class LogSumExp<std::tuple<nntile::fp64_t>>;
template class LogSumExp<std::tuple<nntile::fp32_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class LogSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class LogSumExp<std::tuple<nntile::bf16_t>>;
template class LogSumExp<std::tuple<nntile::fp16_t>>;

//! Pack of logsumexp operations for different types
logsumexp_pack_t logsumexp;

} // namespace nntile::haul
