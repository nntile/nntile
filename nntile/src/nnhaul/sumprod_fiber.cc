/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumprod_fiber.cc
 * Sums over slices into a fiber of a product of two StarPU buffers
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sumprod_fiber.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sumprod_fiber.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked sumprod_fiber driver (chunk_units.hh fiber scheme).
//
// Kernel index map (derived from kernel::sumprod_fiber::cpu/cuda, NOT
// assumed): src1/src2 are [m, k, n] with element (i0, i2, i1) at flat
// i0 + m*(i2 + k*i1) - the contiguous unit is the m-element fiber,
// fibers group into k-fiber planes per i1, and dst is the k-element
// fiber dst[i2]: one accumulator per i2, reduced over the n planes'
// fiber pairs. No batch axis (the fiber operand never re-bases), so
// pieces may span planes freely (one walk panel of k*n fibers).
// split_fiber_piece re-cuts them into (m, P, k, 1) whole-plane and
// (m, 1, L, 1) fiber-offset sub-pieces whose PARTIAL product-sums
// chain through the kernel's OWN beta!=0 accumulate contract: a
// sub-piece starting in plane 0 (every accumulator's first
// contribution is its plane-0 fiber pair) takes the caller's
// (alpha, beta), every continuation takes (alpha, 1) - algebraically
// exact, equal to the plain call up to the rounding of the piecewise
// Kahan merge (the CUDA kernel adds its shared-memory atomicAdd
// nondeterminism, which the plain call already has). The dst fiber
// segment [0, k) is used in place or staged through a bounded k temp
// (gathered when the caller's beta != 0, zero-initialized otherwise);
// straddling src1/src2 fibers are gathered into bounded m temps.
// Zero new kernels; all-unchunked buffers keep the single classic
// call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void sumprod_fiber_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, ::nnhaul::Buffer *buf_src2, Scalar beta,
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
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(k), k * n};
    bool const chunked_src1 = cv_src1.nchunks > 1;
    bool const chunked_src2 = cv_src2.nchunks > 1;
    T *temp_src1 = chunked_src1 ? arena.alloc(std::size_t(m)) : nullptr;
    T *temp_src2 = chunked_src2 ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> dst_stage(
        cv_dst, std::size_t(k), true, arena);
    chunkunits::walk_unit_runs(Index{1}, k * n,
        [&](Index i, Index len, bool /*first*/, bool src1_str, bool src2_str,
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
                    T *pd = dst_stage.advance(Index{0}, beta != Scalar{0})
                        + (plane_span ? 0 : j % k);
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
                    if(p1 == nullptr || p2 == nullptr || pd == nullptr
                        || !arena.ok_state())
                    {
                        return;
                    }
                    Scalar const beta_eff = first_b ? beta : Scalar{1};
                    if(plane_span)
                    {
                        sub(m, L / k, k, p1, p2, beta_eff, pd);
                    }
                    else
                    {
                        sub(m, Index{1}, L, p1, p2, beta_eff, pd);
                    }
                });
        },
        span_src1, span_src2, span_dst);
    dst_stage.retire();
}

} // namespace

//! Constructor
template<typename T>
SumProdFiber<std::tuple<T>>::SumProdFiber():
    codelet(
        "nntile_sumprod_fiber",
        &SumProdFiber<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SumProdFiber<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SumProdFiber<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sumprod_fiber::cpu<T>
template<typename T>
void SumProdFiber<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
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
        chunkunits::classic_dispatch_tripwire("nntile_sumprod_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sumprod_fiber::cpu<T>(args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            args->beta, buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        sumprod_fiber_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src1,
            buf_src2,
            args->beta,
            buf_dst,
            arena,
            [alpha = args->alpha](Index mm, Index nn, Index kk,
                T const *p1, T const *p2, Scalar be, T *pd) {
                kernel::sumprod_fiber::cpu<T>(mm, nn, kk, alpha, p1, p2,
                    be, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_sumprod_fiber: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumProdFiber<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sumprod_fiber::cuda<T>
template<typename T>
void SumProdFiber<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
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
        chunkunits::classic_dispatch_tripwire("nntile_sumprod_fiber", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::sumprod_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(),
            args->beta, buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path (see sum_slice.cc for the rationale): one
    // chunk-aware kernel over the whole logical range; the chunk base-pointer
    // tables are built on the host and parked in the shared device chunk-ptr ring's
    // device-resident ring (no chunk-count cap), and the kernel stages
    // them into shared memory per block. The piece driver stays as the
    // fallback for a failed table allocation.
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
        kernel::sumprod_fiber::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), buf_src2->get_ptr<T>(), args->beta,
            buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1);
    chunkunits::ChunkView<T> const cv_src2(buf_src2);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(args->m > 0 && args->n > 0 && args->k > 0)
    {
            chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_s1_ptrs = tabs.add(cv_src1);
        std::size_t const o_s2_ptrs = tabs.add(cv_src2);
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
                kernel::sumprod_fiber::chunked<T>(
                    stream,
                    args->m,
                    args->n,
                    args->k,
                    args->alpha,
                    t1,
                    Index(cv_src1.nchunks),
                    t2,
                    Index(cv_src2.nchunks),
                    args->beta,
                    td,
                    Index(cv_dst.nchunks),
                    Index(cv_src1.chunk_elems),
                    Index(cv_src2.chunk_elems),
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
        "nntile_sumprod_fiber: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void SumProdFiber<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdFiber<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdFiber<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sumprod_fiber tasks
template<typename T>
std::uint64_t SumProdFiber<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Hash over alpha
    uint32_t hash = args->alpha == Scalar{0} ? -1 : 0;
    // Apply hash over parameters m, n and k
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SumProdFiber<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, Scalar alpha, ::nnhaul::Handle & src1, ::nnhaul::Handle & src2,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert sumprod_fiber task into StarPU pool of tasks
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
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src1_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * k;
    double nflops = beta == 0.0 ? 2*src1_nbytes + dst_nbytes :
        2 * (src1_nbytes+dst_nbytes);
    // Submit task
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked fiber driver; sumprod_fiber never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumProdFiber<std::tuple<nntile::fp64_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumProdFiber<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumProdFiber<std::tuple<nntile::bf16_t>>;
template class SumProdFiber<std::tuple<nntile::fp16_t>>;

//! Pack of sumprod_fiber operations for different types
sumprod_fiber_pack_t sumprod_fiber;

} // namespace nntile::haul
