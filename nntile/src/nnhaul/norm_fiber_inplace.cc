/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_fiber_inplace.cc
 * Euclidean norms over slices into a fiber of a product of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_fiber_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_fiber_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked norm_fiber_inplace driver (chunk_units.hh fiber
// scheme). Same index map and piecewise-hypot chaining as norm_fiber
// (see norm_fiber.cc) with the accumulator base and the output the
// SAME buffer: the first sub-piece of a batch takes the caller's
// (alpha, beta) reading the dst batch segment, every continuation
// takes beta=1 (the kernel then reads its own dst accumulator - the
// alias is implicit). The dst batch segment is used in place or staged
// through a bounded k temp (gathered when the caller's beta != 0,
// zero-initialized otherwise); straddling src fibers are gathered into
// a bounded m temp. Zero new kernels; all-unchunked buffers keep the
// single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void norm_fiber_inplace_chunked(Index m, Index n, Index k, Index batch,
    Scalar alpha, ::nnhaul::Buffer *buf_src, Scalar beta,
    ::nnhaul::Buffer *buf_dst, Arena &arena,
    Sub &&sub)
{
    if(m <= 0 || k <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(k), k * n};
    bool const chunked_src = cv_src.nchunks > 1;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(k), true, arena);
    chunkunits::walk_unit_runs(batch, k * n,
        [&](Index i, Index len, bool /*first*/, bool src_str,
            bool /*dst_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            chunkunits::split_fiber_piece(i, len, n, k,
                [&](Index j, Index L, bool plane_span, bool first_b)
                {
                    if(L <= 0 || !arena.ok_state())
                    {
                        return;
                    }
                    Index const b = j / (n * k);
                    T *pd = dst_stage.advance(b, beta != Scalar{0})
                        + (plane_span ? 0 : j % k);
                    T const *ps = nullptr;
                    if(src_str)
                    {
                        chunkunits::gather_range(cv_src, std::size_t(m) * j,
                            std::size_t(m), temp_src, arena);
                        ps = temp_src;
                    }
                    else
                    {
                        ps = span_src.ptr(j);
                    }
                    if(ps == nullptr || pd == nullptr || !arena.ok_state())
                    {
                        return;
                    }
                    Scalar const beta_eff = first_b ? beta : Scalar{1};
                    if(plane_span)
                    {
                        sub(m, L / k, k, ps, beta_eff, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, ps, beta_eff, pd);
                    }
                });
        },
        span_src, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
NormFiberInplace<std::tuple<T>>::NormFiberInplace():
    codelet(
        "nntile_norm_fiber_inplace",
        &NormFiberInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &NormFiberInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &NormFiberInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::norm_fiber::cpu<T>
template<typename T>
void NormFiberInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        kernel::norm_fiber_inplace::cpu<T>(args->m, args->n, args->k,
            args->batch, args->alpha, buf_src->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        norm_fiber_inplace_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_src,
            args->beta,
            buf_dst,
            arena,
            [alpha = args->alpha](Index mm, Index nn, Index kk,
                T const *ps, Scalar be, T *pd) {
                kernel::norm_fiber_inplace::cpu<T>(mm, nn, kk, Index{1},
                    alpha, ps, be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_norm_fiber_inplace: cpu chunked driver aborted after "
            "an allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormFiberInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::norm_fiber::cuda<T>
template<typename T>
void NormFiberInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_fiber_inplace::cuda<T>(stream, args->m, args->n,
            args->k, args->batch, args->alpha, buf_src->get_ptr<T>(),
            args->beta, buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    norm_fiber_inplace_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        arena,
        [stream, alpha = args->alpha](Index mm, Index nn, Index kk,
            T const *ps, Scalar be, T *pd) {
            kernel::norm_fiber_inplace::cuda<T>(stream, mm, nn, kk,
                Index{1}, alpha, ps, be, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_norm_fiber_inplace: chunked CUDA driver aborted after "
            "a stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void NormFiberInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormFiberInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiberInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_fiber tasks
template<typename T>
std::uint64_t NormFiberInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    hash = ::nntile::haul::fnv1a(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void NormFiberInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index batch, Scalar alpha, ::nnhaul::Handle & src,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert norm_fiber_inplace task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
        }
        else
        {
            dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; norm_fiber_inplace never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormFiberInplace<std::tuple<nntile::fp64_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormFiberInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormFiberInplace<std::tuple<nntile::bf16_t>>;

//! Pack of norm_fiber_inplace operations for different types
norm_fiber_inplace_pack_t norm_fiber_inplace;

} // namespace nntile::haul
