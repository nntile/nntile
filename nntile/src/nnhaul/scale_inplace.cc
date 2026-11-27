/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/scale_inplace.cc
 * Scale inplace operation on a NNHaul buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/scale_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scale_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
ScaleInplace<std::tuple<T>>::ScaleInplace():
    codelet(
        "nntile_scale_inplace",
        &ScaleInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &ScaleInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &ScaleInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! NNHaul wrapper for kernel::scale_inplace::cpu<T>
template<typename T>
void ScaleInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto data = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *p) {
            kernel::scale_inplace::cpu<T>(n, args->alpha, p);
        },
        data);
}

// Specializations of CPU wrapper for accelerated types
template<>
void ScaleInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! NNHaul wrapper for kernel::scale_inplace::cuda<T>
template<typename T>
void ScaleInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_scale_inplace");
    // Get interfaces
    auto *data_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(data_buf->chunk_count() == 1)
    {
        if(args->nelems > 0)
        {
            kernel::scale_inplace::cuda<T>(stream, args->nelems, args->alpha,
                data_buf->get_ptr<T>());
        }
        return;
    }
    if(args->nelems > 0)
    {
        chunkunits::ChunkView<T> const cvd(data_buf);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_pd = tabs.add(cvd);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                kernel::scale_inplace::chunked<T>(stream, args->nelems,
                    args->alpha,
                    arena.table_view<T *>(table_base, o_pd),
                    Index(cvd.nchunks),
                    Index(cvd.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_scale_inplace: chunk-ptr table cannot serve this launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void ScaleInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void ScaleInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    ScaleInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scale_inplace tasks that depends only on cl_arg
template<typename T>
std::uint64_t ScaleInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void ScaleInplace<std::tuple<T>>::submit(int worker_hint,
    Index nelems, Scalar alpha, ::nnhaul::Handle & data)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero, function reduces to clear
    if(alpha == zero)
    {
        clear.submit(worker_hint, data);
        return;
    }
    // if alpha is one, function reduces to no-op
    if(alpha == 1.0)
    {
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_RW, &data } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class ScaleInplace<std::tuple<nntile::fp64_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class ScaleInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class ScaleInplace<std::tuple<nntile::bf16_t>>;
template class ScaleInplace<std::tuple<nntile::fp16_t>>;

//! Pack of scale_inplace operations for different types
scale_inplace_pack_t scale_inplace;

} // namespace nntile::nnhaul
