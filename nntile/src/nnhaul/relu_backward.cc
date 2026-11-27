/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/relu_backward.cc
 * Backward ReLU operation on a NNHaul buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/relu_backward.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/relu_backward.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
ReluBackward<std::tuple<T>>::ReluBackward():
    codelet(
        "nntile_relu_backward",
        &ReluBackward<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &ReluBackward<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &ReluBackward<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::relu_backward::cpu<T>
template<typename T>
void ReluBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto x = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dy = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dx = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *px, const T *pdy, T *pdx) {
            kernel::relu_backward::cpu<T>(n, args->alpha, px, pdy, args->beta, pdx);
        },
        x,
        dy,
        dx);
}

// Specializations of CPU wrapper for accelerated types
template<>
void ReluBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ReluBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ReluBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

//! Define codelet pack
#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::relu_backward::cuda<T>
template<typename T>
void ReluBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_relu_backward");
    // Get interfaces
    auto *x_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *dy_buf = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto *dx_buf = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(x_buf->chunk_count() == 1 && dy_buf->chunk_count() == 1
            && dx_buf->chunk_count() == 1)
    {
        if(args->nelems > 0)
        {
            kernel::relu_backward::cuda<T>(stream, args->nelems,
                args->alpha, x_buf->get_ptr<T>(), dy_buf->get_ptr<T>(),
                args->beta, dx_buf->get_ptr<T>());
        }
        return;
    }
    if(args->nelems > 0)
    {
        chunkunits::ChunkView<T> const cvx(x_buf);
        chunkunits::ChunkView<T> const cvdy(dy_buf);
        chunkunits::ChunkView<T> const cvdx(dx_buf);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_px = tabs.add(cvx);
        std::size_t const o_pdy = tabs.add(cvdy);
        std::size_t const o_pdx = tabs.add(cvdx);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                kernel::relu_backward::chunked<T>(stream, args->nelems,
                    args->alpha,
                    arena.table_view<const T *>(table_base, o_px),
                    Index(cvx.nchunks), Index(cvx.chunk_elems),
                    arena.table_view<const T *>(table_base, o_pdy),
                    Index(cvdy.nchunks), Index(cvdy.chunk_elems),
                    args->beta,
                    arena.table_view<T *>(table_base, o_pdx),
                    Index(cvdx.nchunks), Index(cvdx.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_relu_backward: chunk-ptr table cannot serve this launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void ReluBackward<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ReluBackward<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ReluBackward<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ReluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
std::uint64_t ReluBackward<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void ReluBackward<std::tuple<T>>::submit(int worker_hint, Index nelems, Scalar alpha,
        ::nnhaul::Handle & x, ::nnhaul::Handle & dy, Scalar beta, ::nnhaul::Handle & dx)
{
    data_access_mode dx_mode;
    if(beta == 0.0)
    {
        dx_mode = ACCESS_W;
    }
    else
    {
        dx_mode = ACCESS_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &x }, { ACCESS_R, &dy }, { dx_mode, &dx } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ReluBackward<std::tuple<nntile::fp64_t>>;
template class ReluBackward<std::tuple<nntile::fp32_t>>;
template class ReluBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ReluBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ReluBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ReluBackward<std::tuple<nntile::bf16_t>>;

//! Pack of relu_backward operations for different types
relu_backward_pack_t relu_backward;

} // namespace nntile::nnhaul
