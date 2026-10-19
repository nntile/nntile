/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/norm.cc
 * Euclidean norm of all elements in a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/norm.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/norm.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked norm driver (chunk_units.hh scheme; the whole-tile
// analogue of family 1's norm_slice hypot chain). The kernel index map is
// flat: src[0..nelems) with the scalar dst[0] combining via
// hypot(alpha*norm(piece), beta*dst). The pieces are the src CHUNKS
// themselves - each contiguous, never straddling its own table - applied
// in flat order: the first piece takes (alpha, caller beta), every
// continuation (alpha, 1), and hypot associativity carries the chain
// (dst >= 0 after the first piece, so an all-zero continuation folds to
// the identity). Zero-anchor checked like family 1 did for norm_slice:
// unlike the norm_slice CUDA of that family, the whole-tile CUDA kernel
// ALWAYS writes dst[0] (thread 0 of the one reduction block, both the
// beta==0 and beta!=0 branches), and its shared-memory hypot tree folds
// zeros exactly, so no kernel fix is needed here. The 1-element dst is
// always a single chunk; no staging exists. Zero new kernels; all
// unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

namespace
{

template<typename T, typename Sub>
void norm_chunked(Index nelems, Scalar alpha, ::nnhaul::Buffer *buf_src,
    Scalar beta, ::nnhaul::Buffer *buf_dst,
    Sub &&sub)
{
    chunkunits::ChunkView<T> const cv_src(buf_src);
    T *pd = buf_dst->get_ptr<T>();
    if(nelems <= 0)
    {
        // Preserve the kernel's empty-input contract on dst with one
        // direct call (norm 0 folds through beta exactly as the kernel).
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
Norm<std::tuple<T>>::Norm():
    codelet(
        "nntile_norm",
        &Norm<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Norm<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Norm<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::norm::cpu<T>
template<typename T>
void Norm<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
        kernel::norm::cpu<T>(args->nelems, args->alpha,
            buf_src->get_ptr<T>(), args->beta, buf_dst->get_ptr<T>());
        return;
    }
    norm_chunked<T>(
        args->nelems,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        [](Index n, Scalar al, T const *ps, Scalar be, T *pd) {
            kernel::norm::cpu<T>(n, al, ps, be, pd);
        });
}

// Specializations of CPU wrapper for accelerated types
template<>
void Norm<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Norm<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Norm<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::norm::cuda<T>
template<typename T>
void Norm<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
        kernel::norm::cuda<T>(
            stream,
            args->nelems,
            args->alpha,
            buf_src->get_ptr<T>(),
            args->beta,
            buf_dst->get_ptr<T>()
        );
        return;
    }
    norm_chunked<T>(
        args->nelems,
        args->alpha,
        buf_src,
        args->beta,
        buf_dst,
        [stream](Index n, Scalar al, T const *ps, Scalar be, T *pd) {
            kernel::norm::cuda<T>(stream, n, al, ps, be, pd);
        });
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Norm<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Norm<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Norm<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Norm<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for norm tasks
template<typename T>
std::uint64_t Norm<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
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
void Norm<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src,
        Scalar beta, ::nnhaul::Handle & dst, int redux)
//! Insert norm task into StarPU pool of tasks
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
    // chunked driver; norm never stages.
    (void)nflops;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Norm<std::tuple<nntile::fp64_t>>;
template class Norm<std::tuple<nntile::fp32_t>>;
template class Norm<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Norm<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Norm<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Norm<std::tuple<nntile::bf16_t>>;
template class Norm<std::tuple<nntile::fp16_t>>;

//! Pack of norm operations for different types
norm_pack_t norm;

} // namespace nntile::haul
