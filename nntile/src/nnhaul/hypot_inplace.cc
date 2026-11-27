/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/hypot_inplace.cc
 * hypot_inplace operation on a NNHaul buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot_inplace.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_inplace.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/clear.hh"
#include "nntile/nnhaul/ops/scale_inplace.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
HypotInplace<std::tuple<T>>::HypotInplace():
    codelet(
        "nntile_hypot_inplace",
        &HypotInplace<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &HypotInplace<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &HypotInplace<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot_inplace operation for NNHaul buffers in CPU
template<typename T>
void HypotInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::hypot_inplace::cpu<T>(n, args->alpha, s, args->beta, d);
        },
        src,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot_inplace for NNHaul buffers on CUDA
template<typename T>
void HypotInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_hypot_inplace");
    // Get interfaces
    auto *src_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *dst_buf = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(src_buf->chunk_count() == 1 && dst_buf->chunk_count() == 1)
    {
        if(args->nelems > 0)
        {
            kernel::hypot_inplace::cuda<T>(stream, args->nelems, args->alpha,
                src_buf->get_ptr<T>(), args->beta, dst_buf->get_ptr<T>());
        }
        return;
    }
    if(args->nelems > 0)
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
                kernel::hypot_inplace::chunked<T>(stream, args->nelems,
                    args->alpha,
                    arena.table_view<const T *>(table_base, o_ps),
                    Index(cvs.nchunks),
                    args->beta,
                    arena.table_view<T *>(table_base, o_pd),
                    Index(cvd.nchunks),
                    Index(cvs.chunk_elems), Index(cvd.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_hypot_inplace: chunk-ptr table cannot serve this "
                "launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void HypotInplace<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotInplace<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for hypot_inplace tasks that depends only on cl_arg
template<typename T>
std::uint64_t HypotInplace<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot_inplace task
template<typename T>
void HypotInplace<std::tuple<T>>::submit(int worker_hint,
        Index nelems, Scalar alpha, ::nnhaul::Handle & src, Scalar beta, ::nnhaul::Handle & dst)
//! Insert hypot_inplace task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // If beta is zero this function reduces to scale
    //if(beta == zero)
    //{
    //    throw std::runtime_error("modulus(apha*src) is not yet implemented");
    //    //scale::submit<T>(nelems, alpha, src, dst);
    //    return;
    //}
    // If beta is non-zero and alpha is zero then reduce to scale_inplace
    //if(alpha == zero)
    //{
    //    throw std::runtime_error("modulus_inplace(beta*dst) is not yet "
    //            "implemented");
    //    //scale_inplace::submit<T>(nelems, beta, dst);
    //    return;
    //}
    // Access mode for the dst handle
    enum data_access_mode dst_mode;
    if(beta == one)
    {
        dst_mode = static_cast<data_access_mode>(ACCESS_RW | ACCESS_COMMUTE);
    }
    else if(beta == zero)
    {
        dst_mode = ACCESS_W;
    }
    else
    {
        dst_mode = ACCESS_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src }, { dst_mode, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class HypotInplace<std::tuple<nntile::fp64_t>>;
template class HypotInplace<std::tuple<nntile::fp32_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class HypotInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class HypotInplace<std::tuple<nntile::bf16_t>>;
template class HypotInplace<std::tuple<nntile::fp16_t>>;

//! Pack of hypot_inplace operations for different types
hypot_inplace_pack_t hypot_inplace;

} // namespace nntile::nnhaul
