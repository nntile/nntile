/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_slice.cc
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/add.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/scale_slice.hh"

//! NNHaul wrappers for add_slice operation
namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked add_slice driver (chunk_units.hh scheme, cf. gemm.cc)
//
// Kernel index map (derived from kernel::add_slice::cpu/cuda, NOT
// assumed): dst[i0 + m*i1 + m*k*i2] = alpha*src1[i0 + m*i2] +
// beta*src2[i0 + m*i1 + m*k*i2] - ELEMENTWISE given the src1 slice
// value of the unit (i0, i2), so whole-row pieces reproduce the plain
// call's per-element operations bitwise:
//   * pieces are runs of whole rows (m contiguous elements) inside
//     single chunks of src2 and dst; the src1 read of a piece is its
//     panel's m-element segment, staged through a bounded m temp when
//     it straddles a src1 chunk boundary;
//   * panel-aligned pieces may span whole panels ((m, n', k) sub-calls);
//   * a row straddling a src2 or dst chunk boundary is gathered into a
//     bounded m-element temp, transformed as (m, 1, 1) and scattered
//     back (dst only).
// beta!=0 semantics live inside the kernel and are preserved verbatim
// (src2 is read, dst is pure output). Zero new kernels; all-unchunked
// buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void add_slice_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src1, Scalar beta, ::nnhaul::Buffer *buf_src2,
    ::nnhaul::Buffer *buf_dst, std::size_t force_bytes, Arena &arena,
    Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src1(buf_src1, force_bytes);
    chunkunits::ChunkView<T> const cv_src2(buf_src2, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_src2{&cv_src2, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src1{&cv_src1, std::size_t(m), k};
    bool const chunked_src2 = cv_src2.nchunks > 1 || force_bytes != 0;
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
    T *temp_src2 = chunked_src2 ? arena.alloc(std::size_t(m)) : nullptr;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src1_stage(
        cv_src1, std::size_t(m), false, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool /*first*/, bool src2_str, bool dst_str,
            bool /*src1_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
            }
            T const *p1 = src1_stage.advance(i / k, false);
            T const *p2 = nullptr;
            if(src2_str)
            {
                chunkunits::gather_range(cv_src2, std::size_t(m) * i,
                    std::size_t(m), temp_src2, arena);
                p2 = temp_src2;
            }
            else
            {
                p2 = span_src2.ptr(i);
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
            if(p1 == nullptr || p2 == nullptr || pd == nullptr
                || !arena.ok_state())
            {
                return;
            }
            if(len >= k)
            {
                sub(m, len / k, k, alpha, p1, beta, p2, pd);
            }
            else
            {
                sub(m, Index{1}, len, alpha, p1, beta, p2, pd);
            }
            if(dst_str)
            {
                chunkunits::scatter_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
            }
        },
        span_src2, span_dst, span_src1);
}

} // namespace

//! Constructor
template<typename T>
AddSlice<std::tuple<T>>::AddSlice():
    codelet(
        "nntile_add_slice",
        &AddSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &AddSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &AddSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::add_slice::cpu<T>
template<typename T>
void AddSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        kernel::add_slice::cpu<T>(args->m, args->n, args->k, args->alpha,
            buf_src1->get_ptr<T>(), args->beta, buf_src2->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        add_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src1,
            args->beta,
            buf_src2,
            buf_dst,
            forced,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *p1,
                Scalar be, T const *p2, T *pd) {
                kernel::add_slice::cpu<T>(mm, nn, kk, al, p1, be, p2, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_add_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::add_slice::cuda<T>
template<typename T>
void AddSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        kernel::add_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src1->get_ptr<T>(), args->beta,
            buf_src2->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    add_slice_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->alpha,
        buf_src1,
        args->beta,
        buf_src2,
        buf_dst,
        forced,
        arena,
        [stream](Index mm, Index nn, Index kk, Scalar al, T const *p1,
            Scalar be, T const *p2, T *pd) {
            kernel::add_slice::cuda<T>(stream, mm, nn, kk, al, p1, be, p2, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_add_slice: chunked CUDA driver aborted after a stream "
            "error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AddSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_slice tasks
template<typename T>
std::uint64_t AddSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void AddSlice<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    Scalar beta,
    ::nnhaul::Handle & src2,
    ::nnhaul::Handle & dst
)
{
    // If k is 1, then this operation reduces to add
    if(k == 1)
    {
        add.submit<std::tuple<T>>(starpu_worker_hint, m*n, alpha, src1, beta, src2, dst);
        return;
    }
    // If alpha is zero then reduce to scale
    if(alpha == 0.0)
    {
        scale.submit<std::tuple<T>>(starpu_worker_hint, m * k * n, beta, src2, dst);
        return;
    }
    // If beta is zero then reduce to scale_slice
    if(beta == 0.0)
    {
        scale_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src1, dst);
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
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = m * n * (2*k+1);
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; add_slice never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddSlice<std::tuple<nntile::fp64_t>>;
template class AddSlice<std::tuple<nntile::fp32_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddSlice<std::tuple<nntile::bf16_t>>;
template class AddSlice<std::tuple<nntile::fp16_t>>;

//! Pack of add_slice operations for different types
add_slice_pack_t add_slice;

} // namespace nntile::haul
