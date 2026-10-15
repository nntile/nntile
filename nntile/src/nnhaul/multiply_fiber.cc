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
 * @version 1.1.0
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
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Arena &arena,
    Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1, force_bytes);
    chunkunits::ChunkView<T> const cv_src2(buf_src2, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(k), k * n};
    bool const chunked_src2 = cv_src2.nchunks > 1 || force_bytes != 0;
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
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
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1 && forced == 0)
    {
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
            forced,
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
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src1->chunk_count() == 1 && buf_src2->chunk_count() == 1
        && buf_dst->chunk_count() == 1 && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    multiply_fiber_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->alpha,
        buf_src1,
        buf_src2,
        buf_dst,
        forced,
        arena,
        [stream](Index mm, Index nn, Index kk, Scalar al, T const *p1,
            T const *p2, T *pd) {
            kernel::multiply_fiber::cuda<T>(stream, mm, nn, kk, al, p1, p2,
                pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_multiply_fiber: chunked CUDA driver aborted after a "
            "stream error; dst left unwritten\n");
    }
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
