/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_fiber_inplace.cc
 * StarPU wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_fiber_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_fiber_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"
#include "nntile/nnhaul/ops/scale_fiber.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked add_fiber_inplace driver (chunk_units.hh fiber scheme).
//
// Same kernel index map as add_fiber (dst [m, k, n, batch] with the
// m-element fiber contiguous, src the k-by-batch fiber operand,
// src[i2 + k*b]); the one behavioral difference: the kernel reads dst
// even when beta is zero (NaN propagation contract), so a straddling
// dst fiber is always GATHERED into its bounded m temp before the
// (m, 1, 1, 1) sub-call and scattered back. Zero new kernels;
// all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void add_fiber_inplace_chunked(Index m, Index n, Index k, Index batch,
    Scalar alpha, ::nnhaul::Buffer *buf_src, Scalar beta,
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Arena &arena,
    Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(k), k * n};
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
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
                        // The kernel reads dst even for beta == 0.
                        chunkunits::gather_range(cv_dst, std::size_t(m) * j,
                            std::size_t(m), temp_dst, arena);
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
                        sub(m, L / k, k, alpha, p1, beta, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, alpha, p1, beta, pd);
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
AddFiberInplace<std::tuple<T>>::AddFiberInplace():
    codelet(
        "nntile_add_fiber_inplace",
        &AddFiberInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AddFiberInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AddFiberInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add_fiber_inplace operation on StarPU buffers on CPU
template<typename T>
void AddFiberInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::add_fiber_inplace::cpu<T>(args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        add_fiber_inplace_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_src,
            args->beta,
            buf_dst,
            forced,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *p1,
                Scalar be, T *pd) {
                kernel::add_fiber_inplace::cpu<T>(mm, nn, kk, Index{1}, al,
                    p1, be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_add_fiber_inplace: cpu chunked driver aborted after "
            "an allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddFiberInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add_fiber_inplace operation on StarPU buffer on CUDA
template<typename T>
void AddFiberInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    std::size_t const forced = chunkunits::forced_chunk_bytes();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1
        && forced == 0)
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::add_fiber_inplace::cuda<T>(stream, args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    add_fiber_inplace_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        forced,
        arena,
        [stream](Index mm, Index nn, Index kk, Scalar al, T const *p1,
            Scalar be, T *pd) {
            kernel::add_fiber_inplace::cuda<T>(stream, mm, nn, kk, Index{1},
                al, p1, be, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_add_fiber_inplace: chunked CUDA driver aborted after "
            "a stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AddFiberInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddFiberInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_fiber_inplace tasks
template<typename T>
std::uint64_t AddFiberInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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

//! Submit add_fiber_inplace task
template<typename T>
void AddFiberInplace<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    ::nnhaul::Handle & src,
    Scalar beta,
    ::nnhaul::Handle & dst
)
{
    // If alpha is zero, then this operation reduces to scale_inplace
    if(alpha == 0.0)
    {
        scale_inplace.submit<std::tuple<T>>(starpu_worker_hint, m*n*k*batch, beta, dst);
        return;
    }
    // If beta is zero, then this operation reduces to scale_fiber
    if(beta == 0.0)
    {
        scale_fiber.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, batch, alpha, src, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n * batch;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; add_fiber_inplace never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddFiberInplace<std::tuple<nntile::fp64_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddFiberInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddFiberInplace<std::tuple<nntile::bf16_t>>;
template class AddFiberInplace<std::tuple<nntile::fp16_t>>;

//! Pack of add_fiber_inplace operations for different types
add_fiber_inplace_pack_t add_fiber_inplace;

} // namespace nntile::haul
