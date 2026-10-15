/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scale_slice.cc
 * StarPU wrappers for scaling of a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale_slice.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked scale_slice driver (chunk_units.hh scheme; see the
// index-map derivation in add_slice.cc - the op is ELEMENTWISE given the
// src slice value of the unit, so whole-row pieces reproduce the plain
// call's per-element operations bitwise). dst is pure output: a row
// straddling a dst chunk boundary is written through a bounded m-element
// temp as (m, 1, 1) and scattered back; the src read of a piece is its
// panel's m-element segment, staged through a bounded temp when it
// straddles. Zero new kernels; all-unchunked buffers keep the single
// classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Arena, typename Sub>
void scale_slice_chunked(Index m, Index n, Index k, Scalar alpha,
    ::nnhaul::Buffer *buf_src, ::nnhaul::Buffer *buf_dst,
    std::size_t force_bytes, Arena &arena, Sub &&sub)
{
    if(m <= 0)
    {
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src, force_bytes);
    chunkunits::ChunkView<T> const cv_dst(buf_dst, force_bytes);
    chunkunits::UnitSpan<T> const span_dst{&cv_dst, std::size_t(m), Index{1}};
    chunkunits::UnitSpan<T> const span_src{&cv_src, std::size_t(m), k};
    bool const chunked_dst = cv_dst.nchunks > 1 || force_bytes != 0;
    T *temp_dst = chunked_dst ? arena.alloc(std::size_t(m)) : nullptr;
    chunkunits::SliceStage<T, Arena> src_stage(
        cv_src, std::size_t(m), false, arena);
    chunkunits::walk_unit_runs(n, k,
        [&](Index i, Index len, bool /*first*/, bool dst_str,
            bool /*src_str*/)
        {
            if(len <= 0 || !arena.ok_state())
            {
                return;
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
            T const *ps = src_stage.advance(i / k, false);
            if(pd == nullptr || ps == nullptr || !arena.ok_state())
            {
                return;
            }
            if(len >= k)
            {
                sub(m, len / k, k, alpha, ps, pd);
            }
            else
            {
                sub(m, Index{1}, len, alpha, ps, pd);
            }
            if(dst_str)
            {
                chunkunits::scatter_range(cv_dst, std::size_t(m) * i,
                    std::size_t(m), temp_dst, arena);
            }
        },
        span_dst, span_src);
}

} // namespace

//! Constructor
template<typename T>
ScaleSlice<std::tuple<T>>::ScaleSlice():
    codelet(
        "nntile_scale_slice",
        &ScaleSlice<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &ScaleSlice<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &ScaleSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::scale_slice::cpu<T>
template<typename T>
void ScaleSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        kernel::scale_slice::cpu<T>(args->m, args->n, args->k, args->alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    try
    {
        chunkunits::ArenaCpu<T> arena;
        scale_slice_chunked<T>(
            args->m,
            args->n,
            args->k,
            args->alpha,
            buf_src,
            buf_dst,
            forced,
            arena,
            [](Index mm, Index nn, Index kk, Scalar al, T const *ps, T *pd) {
                kernel::scale_slice::cpu<T>(mm, nn, kk, al, ps, pd);
            });
    }
    catch(...)
    {
        // A noexcept body must not throw; a failed scratch allocation
        // leaves the result unwritten (same failure class as a failed
        // kernel launch).
        std::fprintf(stderr,
            "nntile_scale_slice: cpu chunked driver aborted after an "
            "allocation failure; dst left unwritten\n");
    }
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::scale_slice::cuda<T>
template<typename T>
void ScaleSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
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
        kernel::scale_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ArenaCuda<T> arena(stream);
    scale_slice_chunked<T>(
        args->m,
        args->n,
        args->k,
        args->alpha,
        buf_src,
        buf_dst,
        forced,
        arena,
        [stream](Index mm, Index nn, Index kk, Scalar al, T const *ps,
            T *pd) {
            kernel::scale_slice::cuda<T>(stream, mm, nn, kk, al, ps, pd);
        });
    if(!arena.ok_state())
    {
        std::fprintf(stderr,
            "nntile_scale_slice: chunked CUDA driver aborted after a "
            "stream error; dst left unwritten\n");
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void ScaleSlice<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleSlice<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scale_slice tasks
template<typename T>
std::uint64_t ScaleSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void ScaleSlice<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    ::nnhaul::Handle & src,
    ::nnhaul::Handle & dst
)
//! Insert scale_slice task into StarPU pool of tasks
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
    double nflops = sizeof(T) * m * (k+1) * n;
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked unit driver; scale_slice never stages.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleSlice<std::tuple<nntile::fp64_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleSlice<std::tuple<nntile::bf16_t>>;
template class ScaleSlice<std::tuple<nntile::fp16_t>>;

//! Pack of scale_slice operations for different types
scale_slice_pack_t scale_slice;

} // namespace nntile::haul
