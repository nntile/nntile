/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/multiply_inplace.cc
 * Per-element multiplication of two NNHaul buffers
 *
 * @version 2.0.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/multiply_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
MultiplyInplace<std::tuple<T>>::MultiplyInplace():
    codelet(
        "nntile_multiply_inplace",
        &MultiplyInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &MultiplyInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &MultiplyInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply multiply on NNHaul buffers on CPU
template<typename T>
void MultiplyInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::multiply_inplace::cpu<T>(n, alpha, s, d);
        },
        src,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void MultiplyInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply multiply on NNHaul buffer on CUDA
template<typename T>
void MultiplyInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_multiply_inplace");
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    auto *src_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *dst_buf = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(src_buf->chunk_count() == 1 && dst_buf->chunk_count() == 1)
    {
        if(nelems > 0)
        {
            kernel::multiply_inplace::cuda<T>(stream, nelems, alpha,
                src_buf->get_ptr<T>(), dst_buf->get_ptr<T>());
        }
        return;
    }
    if(nelems > 0)
    {
        chunkunits::ChunkView<T> const cvs(src_buf);
        chunkunits::ChunkView<T> const cvd(dst_buf);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_ps = tabs.add(cvs);
        std::size_t const o_pd = tabs.add(cvd);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                kernel::multiply_inplace::chunked<T>(stream, nelems, alpha,
                    arena.table_view<const T *>(table_base, o_ps),
                    Index(cvs.nchunks),
                    Index(cvs.chunk_elems),
                    arena.table_view<T *>(table_base, o_pd),
                    Index(cvd.nchunks),
                    Index(cvd.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_multiply_inplace: chunk-ptr table cannot serve this launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void MultiplyInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MultiplyInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    MultiplyInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for multiply_inplace operation
template<typename T>
std::uint64_t MultiplyInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::nnhaul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    return hash;
}

//! Submit multiply_inplace operation
template<typename T>
void MultiplyInplace<std::tuple<T>>::submit(int worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { ACCESS_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MultiplyInplace<std::tuple<nntile::fp64_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MultiplyInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MultiplyInplace<std::tuple<nntile::bf16_t>>;
template class MultiplyInplace<std::tuple<nntile::fp16_t>>;

//! Pack of multiply_inplace operations for different types
multiply_inplace_pack_t multiply_inplace;

} // namespace nntile::nnhaul
