/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply_fiber.cc
 * StarPU wrappers for per-element product of a tensor and a broadcasted fiber
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/multiply_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply_fiber.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked multiply_fiber driver (chunk_units.hh fiber scheme, cf.
// add_fiber.cc). Kernel index map (derived from
// kernel::multiply_fiber::cpu/cuda, NOT assumed): src2/dst are [m, k, n]
// with element (i0, i2, i1) at flat i0 + m*(i2 + k*i1) - the contiguous
// unit is the m-element fiber, and src1 is the k-element fiber operand
// src1[i2] shared by EVERY plane (no batch axis: no re-basing, so
// pieces may span planes freely). Pieces are runs of whole fibers
// inside single src2 and dst chunks (one walk panel of k*n fibers);
// split_fiber_piece re-cuts them into (m, P, k, 1) whole-plane and
// (m, 1, L, 1) fiber-offset sub-calls; the src1 segment [0, k) is used
// in place or staged once through a bounded k temp; straddling fibers
// are staged through bounded m temps (src2: gather; dst: pure write).
// Zero new kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void multiply_fiber_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, ::nnhaul::Buffer *buf_src2,
    ::nnhaul::Buffer *buf_dst, Arena &arena,
    Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1);
    chunkunits::ChunkView<T> const cv_src2(buf_src2);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(k), k * n};
    bool const chunked_src2 = cv_src2.nchunks > 1;
    bool const chunked_dst = cv_dst.nchunks > 1;
    T *temp_src2 = chunked_src2 ? arena.alloc(std::size_t(m)) : nullptr;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src1_stage(
        cv_src1, std::size_t(k), false, arena);
    chunkunits::walk_unit_runs(Index{1}, k * n,
        [&](Index i, Index len, bool /*first*/, bool src2_str, bool dst_str,
            bool /*src1_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            chunkunits::split_fiber_piece(i, len, n, k,
                [&](Index j, Index L, bool plane_span, bool /*first_b*/)
                {
                    if(L <= 0 || !arena.ok_state())
                    {
                        return;
                    }
                    T const *p1 = src1_stage.advance(Index{0}, false)
                        + (plane_span ? 0 : j % k);
                    T const *p2 = nullptr;
                    if(src2_str)
                    {
                        chunkunits::gather_range(cv_src2, std::size_t(m) * j,
                            std::size_t(m), temp_src2, arena);
                        p2 = temp_src2;
                    }
                    else
                    {
                        p2 = span_src2.ptr(j);
                    }
                    T *pd = nullptr;
                    if(dst_str)
                    {
                        pd = temp_dst;
                    }
                    else
                    {
                        pd = span_dst.ptr(j);
                    }
                    if(p1 == nullptr || p2 == nullptr || pd == nullptr
                        || !arena.ok_state())
                    {
                        return;
                    }
                    if(plane_span)
                    {
                        sub(m, L / k, k, alpha, p1, p2, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, alpha, p1, p2, pd);
                    }
                    if(dst_str)
                    {
                        chunkunits::scatter_range(cv_dst, std::size_t(m) * j,
                            std::size_t(m), temp_dst, arena);
                    }
                });
        },
        span_src2, span_dst, span_src1);
    src1_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
MultiplyFiber<std::tuple<T>>::MultiplyFiber():
    codelet(
        "nntile_multiply_fiber",
        &MultiplyFiber<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MultiplyFiber<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MultiplyFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::multiply_fiber::cpu<T>
template<typename T>
void MultiplyFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if((buf_src1->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src2->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_multiply_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_fiber::cpu<T>(args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        multiply_fiber_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src1,
            buf_src2,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *p1,
                T const *p2, T *pd) {
                kernel::multiply_fiber::cpu<T>(mm, nn, kk, al, p1, p2, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_multiply_fiber: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void MultiplyFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::multiply_fiber::cuda<T>
template<typename T>
void MultiplyFiber<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src1 = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src2 = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src1->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src2->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_multiply_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: issue the WHOLE region as ONE chunk-aware
    // kernel instead of the per-piece driver's one launch per chunk-resident
    // fiber/plane run plus a gather/scatter staging pair for every fiber that
    // straddles a chunk boundary. The chunk base-pointer tables are built on
    // the host and parked in the shared device chunk-ptr ring (no chunk-count cap); the kernel stages them into
    // shared memory per block. Elementwise and bitwise the whole-buffer call.
    // The per-piece driver below stays as the fallback for a
    // test hook and a failed table allocation.
    // Piece C (2026-10-18c): the all-1-chunk degenerate launch is
    // the FIRST thing the chunked block does -- the chunk counts are
    // read straight off the buffers, before any
    // ChunkView/UnitSpan/RingTables work exists, so the degenerate
    // host path is as lean as the classic dispatch above (gemm
    // path-F precedent; launch pointer-identical, bitwise by
    // construction, ring untouched).
    if(args->m > 0 && args->n > 0 && args->k > 0
        && buf_src1->chunk_count() == 1
        && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        kernel::multiply_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_s1(buf_src1);
    chunkunits::ChunkView<T> const cv_s2(buf_src2);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0)
    {
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_s1_ptrs = tabs.add(cv_s1);
        std::size_t const o_s2_ptrs = tabs.add(cv_s2);
        std::size_t const o_dst_ptrs = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *t1 = arena.table_view<const T *>(
                    ring_base, o_s1_ptrs);
                const T *const *t2 = arena.table_view<const T *>(
                    ring_base, o_s2_ptrs);
                T *const *td = arena.table_view<T *>(ring_base, o_dst_ptrs);
                kernel::multiply_fiber::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    args->alpha,
                    t1,
                    Index(cv_s1.nchunks),
                    Index(cv_s1.chunk_elems),
                    t2,
                    Index(cv_s2.nchunks),
                    Index(cv_s2.chunk_elems),
                    td,
                    Index(cv_dst.nchunks),
                    Index(cv_dst.chunk_elems));
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
        "nntile_multiply_fiber: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void MultiplyFiber<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplyFiber<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplyFiber<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for multiply_fiber tasks
template<typename T>
std::uint64_t MultiplyFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void MultiplyFiber<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, ::nnhaul::Handle & src2,
        ::nnhaul::Handle & dst)
//! Insert multiply_fiber task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; multiply_fiber never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MultiplyFiber<std::tuple<nntile::fp64_t>>;
template class MultiplyFiber<std::tuple<nntile::fp32_t>>;
template class MultiplyFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MultiplyFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MultiplyFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MultiplyFiber<std::tuple<nntile::bf16_t>>;
template class MultiplyFiber<std::tuple<nntile::fp16_t>>;

//! Pack of multiply_fiber operations for different types
multiply_fiber_pack_t multiply_fiber;

} // namespace nntile::haul
