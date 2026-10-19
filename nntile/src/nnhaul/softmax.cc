/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/softmax.cc
 * Softmax operation for StarPU buffer
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/softmax.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/softmax.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{


// ---------------------------------------------------------------------------
// Native chunked softmax driver (chunk_units.hh scheme, cf. gemm.cc)
//
// Kernel index map (derived from kernel::softmax::cpu/cuda, NOT assumed):
// dst[i0 + m*i1 + m*k*i2] = alpha*exp(src[i0 + m*i1 + m*k*i2]
//     - mse[2*(i0+m*i2)]) / mse[2*(i0+m*i2)+1], skipping +-inf src values.
// The op is ELEMENTWISE given the maxsumexp pairs of the unit (i0, i2), so:
//   * pieces are runs of whole rows (m contiguous elements) inside single
//     chunks of src and dst - each output element sees the identical
//     operations as in the plain call, bitwise on CPU and CUDA alike;
//   * a piece's maxsumexp read is its panel's 2*m-element segment
//     [2*m*i2, 2*m*(i2+1)), used in place when it lies in one chunk and
//     staged through a bounded 2*m temp otherwise;
//   * panel-aligned pieces may span whole panels ((m, n', k) sub-calls);
//   * a row straddling a src or dst chunk boundary is gathered into a
//     bounded m-element temp, transformed as (m, 1, 1) and scattered
//     back.
// Zero new kernels; all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void softmax_chunked(Index m, Index n, Index k, ::nnhaul::Buffer *buf_mse,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
    Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_mse(buf_mse);
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_mse{&cv_mse, std::size_t{2} * m, k};
    bool const chunked_src = cv_src.nchunks > 1;
    bool const chunked_dst = cv_dst.nchunks > 1;
    T *temp_src = chunked_src ? arena.alloc(std::size_t(m)) : nullptr;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> mse_stage(
        cv_mse, std::size_t{2} * m, false, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool /*first*/, bool src_str, bool dst_str,
            bool /*mse_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            T const *ps = nullptr;
            if(src_str)
            {
                chunkunits::gather_range(cv_src, std::size_t(m) * i,
                    std::size_t(m), temp_src, arena);
                ps = temp_src;
            }
            else
            {
                ps = span_src.ptr(i);
            }
            T *pd = nullptr;
            if(dst_str)
            {
                pd = temp_dst;
            }
            else
            {
                pd = span_dst.ptr(i);
            }
            T *pm = mse_stage.advance(i / k, false);
            if(ps == nullptr || pd == nullptr || pm == nullptr
                || !arena.ok_state())
            {
                return;
            }
            if(len >= k)
            {
                sub(m, len / k, k, pm, ps, pd);
            }
            else
            {
                sub(m, Index{1}, len, pm, ps, pd);
            }
            if(dst_str)
            {
                chunkunits::scatter_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
            }
        },
        span_src, span_dst, span_mse);
}

} // namespace

//! Constructor
template<typename T>
Softmax<std::tuple<T>>::Softmax():
    codelet(
        "nntile_softmax",
        &Softmax<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Softmax<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Softmax<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::softmax::cpu<T>
template<typename T>
void Softmax<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    if((buf_mse->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_softmax", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::softmax::cpu<T>(args->m, args->n, args->k,
            buf_mse->get_ptr<T>(), buf_src->get_ptr<T>(), args->alpha,
            buf_dst->get_ptr<T>());
        return;
    }
    Scalar const alpha = args->alpha;
    try
    {
        chunkunits::ArenaCpu<T> arena;
        softmax_chunked<T>(
            args->m,
            args->n,
            args->k,
            buf_mse,
            buf_src,
            buf_dst,
            arena,
            [alpha](Index mm, Index nn, Index kk, T const *pm, T const *ps,
                T *pd) {
                kernel::softmax::cpu<T>(mm, nn, kk, pm, ps, alpha, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_softmax: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void Softmax<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Softmax<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Softmax<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::softmax::cuda<T>
template<typename T>
void Softmax<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto buf_mse = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_mse->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
        && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels()))
    {
        // Classic contiguous dispatch: illegal under the force flag
        // (the tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_softmax", false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::softmax::cuda<T>(stream, args->m, args->n, args->k,
            buf_mse->get_ptr<T>(), buf_src->get_ptr<T>(), args->alpha,
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: whenever any operand is chunked, process
    // the WHOLE tensor with ONE chunk-aware kernel. The merged chunk
    // base-pointer table rides the device ring (ONE control-stream copy
    // per launch, event-ordered ahead of the kernel); the kernel reads
    // it per-thread by shift/mask. The elementwise formula is
    // byte-for-byte cuda()'s, so the result is bitwise cuda()'s. (The
    // historical piecewise fallback is DELETED -- rule 11; a launch the
    // ring cannot serve aborts.)
    chunkunits::ChunkView<T> const cv_mse(buf_mse);
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_mse_ptrs = tabs.add(cv_mse);
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        std::size_t const o_dst_ptrs = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *mse_tab = arena.table_view<const T *>(
                    ring_base, o_mse_ptrs);
                const T *const *src_tab = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                T *const *dst_tab = arena.table_view<T *>(
                    ring_base, o_dst_ptrs);
                kernel::softmax::chunked<T>(stream, args->m, args->n,
                    args->k, mse_tab, Index(cv_mse.nchunks), src_tab,
                    Index(cv_src.nchunks), args->alpha, dst_tab,
                    Index(cv_dst.nchunks), Index(cv_mse.chunk_elems),
                    Index(cv_src.chunk_elems), Index(cv_dst.chunk_elems));
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
        "nntile_softmax: chunk-ptr ring cannot serve this launch (invariant "
        "violation; the per-piece staging fallback was deleted by rule "
        "11)\n");
    std::abort();
}


// Specializations of CUDA wrapper for accelerated types
template<>
void Softmax<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Softmax<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Softmax<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    Softmax<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for softmax tasks that depends only on m, n and k
template<typename T>
std::uint64_t Softmax<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k. This way if we swap values of m,
    // n and k, then the total size of buffers will remain the same, but the
    // footprint will be different
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void Softmax<std::tuple<T>>::submit(int starpu_worker_hint, Index m, Index n, Index k, ::nnhaul::Handle & maxsumexp,
        ::nnhaul::Handle & src, Scalar alpha, ::nnhaul::Handle & dst)
//! Insert softmax task into StarPU pool of tasks
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
    args->alpha = alpha;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (2*k+1) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; softmax never goes through the staging
    // fallback.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &maxsumexp }, { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Softmax<std::tuple<nntile::fp64_t>>;
template class Softmax<std::tuple<nntile::fp32_t>>;
template class Softmax<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Softmax<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Softmax<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Softmax<std::tuple<nntile::bf16_t>>;
template class Softmax<std::tuple<nntile::fp16_t>>;

//! Pack of softmax operations for different types
softmax_pack_t softmax;

} // namespace nntile::haul
