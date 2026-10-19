/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/total_sum_accum.cc
 * Total sum accumulating for StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/total_sum_accum.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/total_sum_accum.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{


// ---------------------------------------------------------------------------
// Native chunked total_sum_accum driver (chunk_units.hh scheme; the
// family-1 sum chaining algebra combined with the indexed-column
// decomposition of subtract_indexed_outputs).
//
// Kernel index map (derived from kernel::total_sum_accum::cpu/cuda, NOT
// assumed): src is n_labels-by-n_outputs Fortran with element (l, i) at
// flat l + i*n_labels - the CONTIGUOUS unit is the n_labels-element
// OUTPUT COLUMN - while logsumexp and labels are the small per-column
// operands (one element each per column, read contiguously along the
// piece) and val is the 1-element float accumulator. Every kernel action
// for output i reads only column i of src plus logsumexp[i]/labels[i].
// A run of whole columns [j0, j0+L) is therefore the plain kernel
// sub-call (alpha, n_labels, L, ignore_index, logsumexp + j0,
// src + j0*n_labels, labels + j0, val); piece ranges are cut at
// logsumexp/labels chunk boundaries, and a src column straddling a
// chunk boundary is READ-staged through a bounded n_labels temp (no
// scatter - src is pure input). The pieces chain through the kernel's
// OWN accumulate contract (*val = *val - alpha*c + alpha*sum): each
// sub-call takes the caller's alpha and folds its Kahan partial into
// val in flat column order - algebraically exact, equal to the plain
// call up to the rounding of the piecewise merge (the CUDA kernel adds
// atomicAdd nondeterminism, which the plain call already has). Zero new
// kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void total_sum_accum_chunked(Scalar alpha, Index n_labels, Index n_outputs,
    Index ignore_index, ::nnhaul::Buffer *buf_logsumexp,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_labels,
    ::nnhaul::Buffer *buf_val, Arena &arena,
    Sub &&sub)
{
    if(n_labels <= 0 || n_outputs <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_lse(buf_logsumexp);
    chunkunits::ChunkView<nntile::int64_t> const cv_labels(buf_labels);
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_col = chunked_src ? arena.alloc(std::size_t(n_labels)) : nullptr;
    float *pv = buf_val->get_ptr<float>();
    Index j = 0;
    while(j < n_outputs && arena.ok_state())
    {
        // Cut the piece at the next logsumexp/labels chunk boundary: the
        // sub-call reads logsumexp[j, j+L) and labels[j, j+L) as one
        // contiguous run each.
        Index L = n_outputs - j;
        {
            std::size_t const lf = std::size_t(j);
            std::size_t const lc = cv_lse.chunk_of(lf);
            Index const cap = static_cast<Index>(cv_lse.chunk_end(lc) - lf);
            if(cap < L)
            {
                L = cap;
            }
            std::size_t const bc = cv_labels.chunk_of(lf);
            Index const bcap = static_cast<Index>(cv_labels.chunk_end(bc)
                - lf);
            if(bcap < L)
            {
                L = bcap;
            }
        }
        std::size_t const f = std::size_t(j) * n_labels;
        T const *psrc = nullptr;
        if(!chunkunits::in_one_chunk(cv_src, f, std::size_t(n_labels)))
        {
            // Straddling column: read-stage it whole through the temp;
            // the sub-call is exactly this one column.
            chunkunits::gather_range(cv_src, f, std::size_t(n_labels),
                temp_col, arena);
            L = 1;
            psrc = temp_col;
        }
        else
        {
            std::size_t const c = cv_src.chunk_of(f);
            Index cap = static_cast<Index>((cv_src.chunk_end(c) - f)
                / n_labels);
            if(cap < L)
            {
                L = cap;
            }
            if(L < 1)
            {
                L = 1;
            }
            psrc = cv_src.ptr(c, f - cv_src.chunk_begin(c));
        }
        if(psrc == nullptr || !arena.ok_state())
        {
            return;
        }
        sub(alpha, n_labels, L, ignore_index,
            cv_lse.ptr(cv_lse.chunk_of(j),
                std::size_t(j) - cv_lse.chunk_begin(cv_lse.chunk_of(j))),
            psrc,
            cv_labels.ptr(cv_labels.chunk_of(j),
                std::size_t(j) - cv_labels.chunk_begin(cv_labels.chunk_of(j))),
            pv);
        j += L;
    }
}

} // namespace

//! Constructor
template<typename T>
TotalSumAccum<std::tuple<T>>::TotalSumAccum():
    codelet(
        "nntile_total_sum_accum",
        &TotalSumAccum<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &TotalSumAccum<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &TotalSumAccum<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::total_sum_accum::cpu<T>
template<typename T>
void TotalSumAccum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Scalar alpha = args->alpha;
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto buf_logsumexp = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_labels = static_cast<::nnhaul::Buffer *>(buffers[2]);
    auto buf_val = static_cast<::nnhaul::Buffer *>(buffers[3]);
    if((buf_logsumexp->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_labels->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_val->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_total_sum_accum", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::total_sum_accum::cpu<T>(alpha, n_labels, n_outputs,
            ignore_index, buf_logsumexp->get_ptr<T>(), buf_src->get_ptr<T>(),
            buf_labels->get_ptr<nntile::int64_t>(), buf_val->get_ptr<float>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        total_sum_accum_chunked<T>(
            alpha,
            n_labels,
            n_outputs,
            ignore_index,
            buf_logsumexp,
            buf_src,
            buf_labels,
            buf_val,
            arena,
            [](Scalar al, Index nl, Index no, Index ig, T const *plse,
                T const *ps, nntile::int64_t const *pb, float *pv) {
                kernel::total_sum_accum::cpu<T>(al, nl, no, ig, plse, ps, pb,
                    pv);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the accumulator partially folded (same failure class as
        // a failed kernel launch).
        std::fprintf(stderr,
            "nntile_total_sum_accum: cpu chunked driver aborted after an "
            "allocation failure; val left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void TotalSumAccum<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::total_sum_accum::cuda<T>
template<typename T>
void TotalSumAccum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Scalar alpha = args->alpha;
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto buf_logsumexp = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_labels = static_cast<::nnhaul::Buffer *>(buffers[2]);
    auto buf_val = static_cast<::nnhaul::Buffer *>(buffers[3]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_logsumexp->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_labels->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_val->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_total_sum_accum", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::total_sum_accum::cuda<T>(stream, alpha, n_labels, n_outputs,
            ignore_index, buf_logsumexp->get_ptr<T>(), buf_src->get_ptr<T>(),
            buf_labels->get_ptr<nntile::int64_t>(), buf_val->get_ptr<float>());
        return;
    }
    // Chunked single-launch path: whenever any operand is chunked, do the
    // WHOLE op with ONE chunk-aware kernel. The chunk base-pointer arrays are
    // rides the device ring (ONE event-ordered control-stream copy per
    // launch) and the kernel reads it per-thread by shift/mask. Each
    // thread's per-column reads and the block-shared float atomicAdd
    // grouping are identical to cuda()'s, so the result is cuda()'s up
    // to the same float-atomic ordering nondeterminism. (The historical
    // piecewise fallback is DELETED -- rule 11.)
    chunkunits::ChunkView<T> const cv_lse(buf_logsumexp);
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<nntile::int64_t> const cv_lab(buf_labels);
    chunkunits::ChunkView<float> const cv_val(buf_val);
    // Piece C (2026-10-18c) smalls-direct fast path: when lse, labels
    // and val are all SINGLE-CHUNK (the graph shape beside the
    // multi-chunk src), their resolved bases travel as plain kernel
    // parameters -- so ONLY the src table is built, uploaded and read
    // through the ring. This halves the host table work and the upload
    // payload of the latency-bound 4-block launch, and removes one
    // dependent ring-slot load per operand per thread from the serial
    // chain. The general four-table form below (and the piecewise
    // fallback) keeps every other shape.
    if(cv_lse.nchunks == 1 && cv_lab.nchunks == 1 && cv_val.nchunks == 1)
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *src_tab = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                kernel::total_sum_accum::chunked<T>(stream, alpha, n_labels,
                    n_outputs, ignore_index, nullptr, 1, src_tab,
                    Index(cv_src.nchunks), nullptr, 1, nullptr, 1, 1,
                    Index(cv_src.chunk_elems), 1, 1,
                    cv_lse.ptr(0, 0), cv_lab.ptr(0, 0),
                    cv_val.ptr(0, 0));
                arena.end(stream);
                return;
            }
        }
    }
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_lse_ptrs = tabs.add(cv_lse);
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        std::size_t const o_lab_ptrs = tabs.add(cv_lab);
        std::size_t const o_val_ptrs = tabs.add(cv_val);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *lse_tab = arena.table_view<const T *>(
                    ring_base, o_lse_ptrs);
                const T *const *src_tab = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                const nntile::int64_t *const *lab_tab = arena.table_view<const nntile::int64_t *>(
                    ring_base, o_lab_ptrs);
                float *const *val_tab = arena.table_view<float *>(
                    ring_base, o_val_ptrs);
                // Piece C: the single-chunk small operands (lse,
                // labels, val -- the graph shape) also hand their
                // resolved bases straight to the kernel, the per-operand
                // degenerate resolution; only a multi-chunk src still
                // reads the ring table.
                kernel::total_sum_accum::chunked<T>(stream, alpha, n_labels,
                    n_outputs, ignore_index, lse_tab, Index(cv_lse.nchunks),
                    src_tab, Index(cv_src.nchunks), lab_tab,
                    Index(cv_lab.nchunks), val_tab, Index(cv_val.nchunks),
                    Index(cv_lse.chunk_elems), Index(cv_src.chunk_elems),
                    Index(cv_lab.chunk_elems), Index(cv_val.chunk_elems),
                    cv_lse.nchunks == 1 ? cv_lse.ptr(0, 0) : nullptr,
                    cv_lab.nchunks == 1 ? cv_lab.ptr(0, 0) : nullptr,
                    cv_val.nchunks == 1 ? cv_val.ptr(0, 0) : nullptr);
                arena.end(stream);
                return;
            }
        }
    }
        // (rule 11, user directive 2026-10-19) There is no per-piece
    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
    // serve any launch up to a 16 TiB operand, so a failed upload is an
    // invariant violation, not a mode to stage around.
    std::fprintf(stderr,
        "nntile_total_sum_accum: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void TotalSumAccum<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for total_sum_accum tasks
template<typename T>
std::uint64_t TotalSumAccum<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = ::nntile::haul::fnv1a(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

template<typename T>
void TotalSumAccum<std::tuple<T>>::submit(int starpu_worker_hint, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index,
            ::nnhaul::Handle & logsumexp, ::nnhaul::Handle & src, ::nnhaul::Handle & class_labels, ::nnhaul::Handle & val)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->alpha = alpha;
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->ignore_index = ignore_index;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver; total_sum_accum never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &logsumexp }, { STARPU_R, &src }, { STARPU_R, &class_labels }, { STARPU_RW | STARPU_COMMUTE, &val } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class TotalSumAccum<std::tuple<nntile::fp64_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_tf32_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_fp16_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_bf16_t>>;
template class TotalSumAccum<std::tuple<nntile::bf16_t>>;
template class TotalSumAccum<std::tuple<nntile::fp16_t>>;

//! Pack of total_sum_accum operations for different types
total_sum_accum_pack_t total_sum_accum;

} // namespace nntile::haul
