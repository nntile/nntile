/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sum.cc
 * Sum all elements of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/sum.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sum.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked sum driver (chunk_units.hh scheme). The kernel index map
// is flat: src[0..nelems) with the scalar dst[0] accumulating the Kahan
// sum through its own beta contract (dst = beta*dst - alpha*c + alpha*sum,
// the sum_slice chaining algebra). The pieces are therefore the src CHUNKS
// themselves - each is a contiguous run that can never straddle a boundary
// of its own chunk table - applied in flat order on the single in-order
// stream: the first piece takes (alpha, caller beta), every continuation
// (alpha, 1). Algebraically exact, equal to the plain call up to the
// rounding of the piecewise Kahan merge (the family-1 quality note; the
// CUDA kernel is deterministic per launch shape). The 1-element dst is
// always a single chunk, so no staging exists at all. Zero new kernels;
// all-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Sub>
void sum_chunked(Index nelems, Scalar alpha, ::nnhaul::Buffer *buf_src,
    Scalar beta, ::nnhaul::Buffer *buf_dst, std::size_t force_bytes,
    Sub &&sub)
{
    chunkunits::ChunkView<T> const cv_src(buf_src, force_bytes);
    T *pd = buf_dst->get_ptr<T>();
    if(nelems <= 0)
    {
        // Preserve the kernel's empty-input contract on dst (beta fold of
        // the zero sum) with one direct call.
        sub(Index{0}, alpha, cv_src.total_elems > 0 ? cv_src.ptr(0, 0)
                                                    : nullptr,
            beta, pd);
        return;
    }
    std::size_t const limit = static_cast<std::size_t>(nelems)
        < cv_src.total_elems ? static_cast<std::size_t>(nelems)
                             : cv_src.total_elems;
    std::size_t done = 0;
    bool first = true;
    while(done < limit)
    {
        std::size_t const c = cv_src.chunk_of(done);
        std::size_t const len =
            limit - done < cv_src.chunk_end(c) - done ? limit - done
                                                      : cv_src.chunk_end(c)
                - done;
        sub(static_cast<Index>(len), alpha, cv_src.ptr(c, 0),
            first ? beta : Scalar{1}, pd);
        done += len;
        first = false;
    }
}

} // namespace

//! Constructor
template<typename T>
Sum<std::tuple<T>>::Sum():
    codelet(
        "nntile_sum",
        &Sum<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Sum<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Sum<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::sum::cpu<T>
template<typename T>
void Sum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        kernel::sum::cpu<T>(args->nelems, args->alpha,
            buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
        return;
    }
    sum_chunked<T>(
        args->nelems,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        forced,
        [](Index n, Scalar al, T const *ps, Scalar be, T *pd) {
            kernel::sum::cpu<T>(n, al, ps, be, pd);
        });
}

// Specializations of CPU wrapper for accelerated types
template<>
void Sum<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::sum::cuda<T>
template<typename T>
void Sum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        kernel::sum::cuda<T>(
            stream,
            args->nelems,
            args->alpha,
            buf_src->get_ptr<T>(),
            args->beta,
            buf_dst->get_ptr<T>()
        );
        return;
    }
    sum_chunked<T>(
        args->nelems,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        forced,
        [stream](Index n, Scalar al, T const *ps, Scalar be, T *pd) {
            kernel::sum::cuda<T>(stream, n, al, ps, be, pd);
        });
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Sum<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sum tasks
template<typename T>
std::uint64_t Sum<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters nelems
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::haul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void Sum<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert sum task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Set destination access mode
    starpu_data_access_mode dst_mode;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * (nelems+1);
    // Submit task: the BODY picks the unchunked fast path or the native
    // chunked driver; sum never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Sum<std::tuple<nntile::fp64_t>>;
template class Sum<std::tuple<nntile::fp32_t>>;
template class Sum<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Sum<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Sum<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Sum<std::tuple<nntile::bf16_t>>;
template class Sum<std::tuple<nntile::fp16_t>>;

//! Pack of sum operations for different types
sum_pack_t sum;

} // namespace nntile::haul
