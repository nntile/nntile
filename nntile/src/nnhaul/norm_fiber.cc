/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm_fiber.cc
 * Euclidean norms over slices into a fiber of a product of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm_fiber.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked norm_fiber driver (chunk_units.hh fiber scheme, cf. the
// norm_slice reduction driver).
//
// Kernel index map (derived from kernel::norm_fiber::cpu/cuda, NOT
// assumed): src1 is [m, k, n, batch] with element (i0, i2, i1, b) at
// flat i0 + m*(i2 + k*(i1 + n*b)) - the contiguous unit is the
// m-element fiber, fibers group into k-fiber planes per (i1, b), and
// the fiber-sized operands are src2/dst[i2 + k*b]: one accumulator per
// (i2, b), reduced over the n planes' fibers. Pieces are runs of whole
// fibers inside single src1 chunks (walk_unit_runs, panels = batches);
// split_fiber_piece re-cuts them into plane/batch sub-pieces whose
// PARTIAL norms chain through the kernel's OWN hypot structure: a
// sub-piece starting in plane 0 of its batch takes the caller's
// (alpha, beta) against the real src2 batch segment, every
// continuation takes beta=1 with src2 ALIASED to the accumulating dst
// segment (the kernel reads src2[x] before writing dst[x],
// element-wise, so the alias is safe on both bodies) - i.e.
// dst <- hypot(dst_partial, alpha*norm_piece) and hypot(hypot(a,b),c)
// = hypot(a,b,c) makes the chain algebraically exact, applied in
// fiber-position order on the single in-order stream. All-zero pieces
// anchor the chain: both bodies write the beta-combined value
// unconditionally (0 for beta==0, fabs(beta*old) otherwise), so zero
// inputs stay legal chunked inputs (family 1's zero-region lesson).
// The dst batch segment ([k*b, k*b+k)) is used in place or staged
// through a bounded k temp for the batch's whole chain (gathered when
// the caller's beta != 0, zero-initialized otherwise); the src2 batch
// segment likewise through a read-only bounded k temp; sub-pieces
// address their sub-range by adding the in-plane fiber offset.
// Straddling src1 fibers are gathered into a bounded m temp and
// reduced as (m, 1, 1, 1) pieces. Rounding differs from the plain
// call by the piecewise hypot chain. Zero new kernels; all-unchunked
// buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void norm_fiber_chunked(Index m, Index n, Index k, Index batch, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, Scalar beta, ::nnhaul::Buffer *buf_src2,
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
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(k), k * n};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(k), k * n};
    bool const chunked_src1 = cv_src1.nchunks > 1;
    T *temp_src1 = chunked_src1 ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src2_stage(
        cv_src2, std::size_t(k), false, arena);
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(k), true, arena);
    chunkunits::walk_unit_runs(batch, k * n,
        [&](Index i, Index len, bool /*first*/, bool src1_str,
            bool /*src2_str*/, bool /*dst_str*/)
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
                    Index const off = plane_span ? 0 : j % k;
                    T *pd = dst_stage.advance(b, beta != Scalar{0}) + off;
                    T const *p1 = nullptr;
                    if(src1_str)
                    {
                        chunkunits::gather_range(cv_src1, std::size_t(m) * j,
                            std::size_t(m), temp_src1, arena);
                        p1 = temp_src1;
                    }
                    else
                    {
                        p1 = span_src1.ptr(j);
                    }
                    T const *p2 = first_b ? src2_stage.advance(b, false) + off
                                          : pd;
                    if(p1 == nullptr || p2 == nullptr || pd == nullptr
                        || !arena.ok_state())
                    {
                        return;
                    }
                    Scalar const beta_eff = first_b ? beta : Scalar{1};
                    if(plane_span)
                    {
                        sub(m, L / k, k, p1, beta_eff, p2, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, p1, beta_eff, p2, pd);
                    }
                });
        },
        span_src1, span_src2, span_dst);
    src2_stage.retire();
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
NormFiber<std::tuple<T>>::NormFiber():
    codelet(
        "nntile_norm_fiber",
        &NormFiber<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &NormFiber<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &NormFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::norm_fiber::cpu<T>
template<typename T>
void NormFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_fiber::cpu<T>(args->m, args->n, args->k, args->batch,
            args->alpha, buf_src1->get_ptr<T>(), args->beta,
            buf_src2->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        norm_fiber_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->batch,
            args->alpha,
            buf_src1,
            args->beta,
            buf_src2,
            buf_dst,
            arena,
            [alpha = args->alpha](Index mm, Index nn, Index kk,
                T const *p1, Scalar be, T const *p2, T *pd) {
                kernel::norm_fiber::cpu<T>(mm, nn, kk, Index{1}, alpha, p1,
                    be, p2, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_norm_fiber: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void NormFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void NormFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::norm_fiber::cuda<T>
template<typename T>
void NormFiber<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::norm_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->batch, args->alpha, buf_src1->get_ptr<T>(), args->beta,
            buf_src2->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    norm_fiber_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->batch,
        args->alpha,
        buf_src1,
        args->beta,
        buf_src2,
        buf_dst,
        arena,
        [stream, alpha = args->alpha](Index mm, Index nn, Index kk,
            T const *p1, Scalar be, T const *p2, T *pd) {
            kernel::norm_fiber::cuda<T>(stream, mm, nn, kk, Index{1}, alpha,
                p1, be, p2, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_norm_fiber: chunked CUDA driver aborted after a stream "
            "error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void NormFiber<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormFiber<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void NormFiber<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    NormFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm_fiber tasks
template<typename T>
std::uint64_t NormFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void NormFiber<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Index batch, Scalar alpha, ::nnhaul::Handle & src1,
        Scalar beta, ::nnhaul::Handle & src2, ::nnhaul::Handle & dst, int redux)
//! Insert norm_fiber task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->batch = batch;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; norm_fiber never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class NormFiber<std::tuple<nntile::fp64_t>>;
template class NormFiber<std::tuple<nntile::fp32_t>>;
template class NormFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class NormFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class NormFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class NormFiber<std::tuple<nntile::bf16_t>>;

//! Pack of norm_fiber operations for different types
norm_fiber_pack_t norm_fiber;

} // namespace nntile::haul
