/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subtract_indexed_outputs.cc
 * Subtract a given value from certain matrix elements for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/subtract_indexed_outputs.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subtract_indexed_outputs.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked subtract_indexed_outputs driver (chunk_units.hh scheme;
// the indexed-scatter column decomposition, the same arena pattern the
// embedding_backward promotion uses for its vocab rows).
//
// Kernel index map (derived from kernel::subtract_indexed_outputs::
// cpu/cuda, NOT assumed): dst is n_labels-by-n_outputs Fortran with
// element (l, i) at flat l + i*n_labels - the CONTIGUOUS unit is the
// n_labels-element OUTPUT COLUMN, and every kernel action on output i
// (the labels[i]==ignore_index column memset, or the single
// dst[labels[i], i] -= val update) touches ONLY column i. The labels
// vector is the small operand, one element per column. A run of whole
// columns [j0, j0+L) is therefore the plain kernel sub-call
// (n_labels, L, ignore_index, val, labels + j0, dst + j0*n_labels);
// the labels range is cut at label-chunk boundaries (a one-column piece
// reads a single label, always chunk-resident), and a dst column
// straddling a chunk boundary is staged through a bounded n_labels temp
// (gathered - the update reads dst[labels[i]] first - kernel
// (n_labels, 1) on the temp, scattered back). Column-local updates, so
// the piecewise result is BITWISE the plain call under any
// factorization. Zero new kernels; all-unchunked buffers keep the
// single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void subtract_indexed_outputs_chunked(Index n_labels, Index n_outputs,
    Index ignore_index, Scalar val, ::nnhaul::Buffer *buf_labels,
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Arena &arena,
    Sub &&sub)
{
    if(n_labels <= 0 || n_outputs <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::ChunkView<nntile::int64_t> const cv_labels(buf_labels,
        force_bytes);
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
    T *temp_col = chunked_dst ? arena.alloc(std::size_t(n_labels)) : nullptr;
    Index j = 0;
    while(j < n_outputs && arena.ok_state())
    {
        // Cut the piece at the next labels-chunk boundary first: the
        // sub-call reads labels[j, j+L) as one contiguous run.
        Index L = n_outputs - j;
        {
            std::size_t const lf = std::size_t(j);
            std::size_t const lc = cv_labels.chunk_of(lf);
            Index const lcap = static_cast<Index>(cv_labels.chunk_end(lc)
                - lf);
            if(lcap < L)
            {
                L = lcap;
            }
        }
        std::size_t const f = std::size_t(j) * n_labels;
        nntile::int64_t const *pl = cv_labels.ptr(cv_labels.chunk_of(j),
            std::size_t(j) - cv_labels.chunk_begin(cv_labels.chunk_of(j)));
        if(!chunkunits::in_one_chunk(cv_dst, f, std::size_t(n_labels)))
        {
            // Straddling column: stage it whole through the bounded temp.
            chunkunits::gather_range(cv_dst, f, std::size_t(n_labels),
                temp_col, arena);
            if(arena.ok_state())
            {
                sub(n_labels, Index{1}, ignore_index, val, pl, temp_col);
                chunkunits::scatter_range(cv_dst, f, std::size_t(n_labels),
                    temp_col, arena);
            }
            ++j;
            continue;
        }
        std::size_t const c = cv_dst.chunk_of(f);
        std::size_t const end = cv_dst.chunk_end(c);
        Index cap = static_cast<Index>((end - f) / n_labels);
        if(cap < L)
        {
            L = cap;
        }
        if(L < 1)
        {
            L = 1;
        }
        sub(n_labels, L, ignore_index, val, pl,
            cv_dst.ptr(c, f - cv_dst.chunk_begin(c)));
        j += L;
    }
}

} // namespace

//! Constructor
template<typename T>
SubtractIndexedOutputs<std::tuple<T>>::SubtractIndexedOutputs():
    codelet(
        "nntile_subtract_indexed_outputs",
        &SubtractIndexedOutputs<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SubtractIndexedOutputs<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SubtractIndexedOutputs<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    Scalar val = args->value;
    // Get interfaces
    auto buf_labels = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_labels->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::subtract_indexed_outputs::cpu<T>(n_labels, n_outputs,
            ignore_index, val, buf_labels->get_ptr<nntile::int64_t>(),
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        subtract_indexed_outputs_chunked<T>(
            n_labels,
            n_outputs,
            ignore_index,
            val,
            buf_labels,
            buf_dst,
            forced,
            arena,
            [](Index nl, Index no, Index ig, Scalar v,
                nntile::int64_t const *pl, T *pd) {
                kernel::subtract_indexed_outputs::cpu<T>(nl, no, ig, v, pl,
                    pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result partially unwritten (same failure class as a
        // failed kernel launch).
        std::fprintf(stderr,
            "nntile_subtract_indexed_outputs: cpu chunked driver aborted "
            "after an allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply subtract_indexed_outputs operation on StarPU buffer on CUDA
template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto buf_labels = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_labels->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::subtract_indexed_outputs::cuda<T>(stream, n_labels,
            n_outputs, ignore_index, args->value,
            buf_labels->get_ptr<nntile::int64_t>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    subtract_indexed_outputs_chunked<T>(
        n_labels,
        n_outputs,
        ignore_index,
        args->value,
        buf_labels,
        buf_dst,
        forced,
        arena,
        [stream](Index nl, Index no, Index ig, Scalar v,
            nntile::int64_t const *pl, T *pd) {
            kernel::subtract_indexed_outputs::cuda<T>(stream, nl, no, ig, v,
                pl, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_subtract_indexed_outputs: chunked CUDA driver aborted "
            "after a stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for subtract_indexed_outputs tasks
template<typename T>
std::uint64_t SubtractIndexedOutputs<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = ::nntile::haul::fnv1a(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::submit(
        Index n_labels, Index n_outputs, Index ignore_index,
            Scalar val, ::nnhaul::Handle & labels, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(args_t));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->value = val;
    args->ignore_index = ignore_index;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked column driver; subtract_indexed_outputs never stages.
    ::nntile::haul::insert_task(
        codelet.raw,
        -1,
        {{STARPU_R, &labels}, {STARPU_RW, &dst}},
        args,
        sizeof(*args));
    std::free(args);
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SubtractIndexedOutputs<std::tuple<nntile::fp64_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::bf16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp16_t>>;

//! Pack of subtract_indexed_outputs operations for different types
subtract_indexed_outputs_pack_t subtract_indexed_outputs;

} // namespace nntile::haul
