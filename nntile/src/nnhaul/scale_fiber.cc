/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_fiber.cc
 * StarPU wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale_fiber.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked scale_fiber driver (chunk_units.hh fiber scheme, cf.
// add_fiber.cc). Same kernel index map: dst [m, k, n, batch] with the
// m-element fiber contiguous, src the k-by-batch fiber operand
// (src[i2 + k*b]); pieces are runs of whole dst fibers inside single
// dst chunks (walk_unit_runs over fibers, panels = batches);
// split_fiber_piece re-cuts them into (m, P, k, 1) whole-plane and
// (m, 1, L, 1) fiber-offset sub-calls; the src batch segment is used
// in place or staged through a bounded k temp; a straddling dst fiber
// is written through a bounded m temp and scattered back (dst is pure
// output, no gather). Zero new kernels; all-unchunked buffers keep
// the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void scale_fiber_chunked(Index m, Index n, Index k, Index batch,
    Scalar alpha, ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(k), k * n};
    bool const chunked_dst = cv_dst.nchunks > 1;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src_stage(
        cv_src, std::size_t(k), false, arena);
    chunkunits::walk_unit_runs(batch, k * n,
        [&](Index i, Index len, bool /*first*/, bool dst_str,
            bool /*src_str*/)
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
                    T const *p1 = src_stage.advance(
                        j / (n * k), false) + (plane_span ? 0 : j % k);
                    T *pd = nullptr;
                    if(dst_str)
                    {
                        pd = temp_dst;
                    }
                    else
                    {
                        pd = span_dst.ptr(j);
                    }
                    if(p1 == nullptr || pd == nullptr || !arena.ok_state())
                    {
                        return;
                    }
                    if(plane_span)
                    {
                        sub(m, L / k, k, alpha, p1, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, alpha, p1, pd);
                    }
                    if(dst_str)
                    {
                        chunkunits::scatter_range(cv_dst, std::size_t(m) * j,
                            std::size_t(m), temp_dst, arena);
                    }
                });
        },
        span_dst, span_src);
    src_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
ScaleFiber<std::tuple<T>>::ScaleFiber():
    codelet(
        "nntile_scale_fiber",
        &ScaleFiber<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &ScaleFiber<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &ScaleFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply scale_fiber operation on StarPU buffers on CPU
template<typename T>
void ScaleFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::scale_fiber::cpu<T>(args->m, args->n, args->k, args->batch,
            args->alpha, buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        scale_fiber_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_src,
            buf_dst,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *p1, T *pd) {
                kernel::scale_fiber::cpu<T>(mm, nn, kk, Index{1}, al, p1,
                    pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_scale_fiber: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply scale_fiber operation on StarPU buffer on CUDA
template<typename T>
void ScaleFiber<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::scale_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    scale_fiber_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        buf_src,
        buf_dst,
        arena,
        [stream](Index mm, Index nn, Index kk, Scalar al, T const *p1,
            T *pd) {
            kernel::scale_fiber::cuda<T>(stream, mm, nn, kk, Index{1}, al,
                p1, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_scale_fiber: chunked CUDA driver aborted after a "
            "stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void ScaleFiber<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleFiber<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scale_fiber tasks
template<typename T>
std::uint64_t ScaleFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

//! Submit scale_fiber task
template<typename T>
void ScaleFiber<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    ::nnhaul::Handle & src,
    ::nnhaul::Handle & dst
)
{
    // Reduce to clear buffer if alpha is zero
    if(alpha == 0.0)
    {
        clear.submit(starpu_worker_hint, dst);
        return;
    }
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * batch * k * m * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; scale_fiber never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleFiber<std::tuple<nntile::fp64_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleFiber<std::tuple<nntile::bf16_t>>;
template class ScaleFiber<std::tuple<nntile::fp16_t>>;

//! Pack of scale_fiber operations for different types
scale_fiber_pack_t scale_fiber;

} // namespace nntile::haul
