/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/hypot.cc
 * hypot operation on a NNHaul buffers
 *
 * @version 2.0.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot.hh"

// Standard libraries
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot.hh"
#include "nntile/nnhaul/chunk_units.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/clear.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
Hypot<std::tuple<T>>::Hypot():
    codelet(
        "nntile_hypot",
        &Hypot<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &Hypot<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &Hypot<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot operation for NNHaul buffers in CPU
template<typename T>
void Hypot<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto src1 = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    auto src2 = ::nntile::nnhaul::buf_chunks<T>(buffers, 1);
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 2);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, const T *s1, const T *s2, T *d) {
            kernel::hypot::cpu<T>(n, args->alpha, s1, args->beta, s2, d);
        },
        src1,
        src2,
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot for NNHaul buffers on CUDA
template<typename T>
void Hypot<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems), "nntile_hypot");
    // Get interfaces
    auto *src1_buf = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto *src2_buf = static_cast<::nnhaul::Buffer *>(buffers[1]);
    auto *dst_buf = static_cast<::nnhaul::Buffer *>(buffers[2]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(src1_buf->chunk_count() == 1 && src2_buf->chunk_count() == 1
            && dst_buf->chunk_count() == 1)
    {
        if(args->nelems > 0)
        {
            kernel::hypot::cuda<T>(stream, args->nelems, args->alpha,
                src1_buf->get_ptr<T>(), args->beta,
                src2_buf->get_ptr<T>(), dst_buf->get_ptr<T>());
        }
        return;
    }
    if(args->nelems > 0)
    {
        chunkunits::ChunkView<T> const cv1(src1_buf);
        chunkunits::ChunkView<T> const cv2(src2_buf);
        chunkunits::ChunkView<T> const cvd(dst_buf);
        chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
        tabs.clear();
        std::size_t const o_p1 = tabs.add(cv1);
        std::size_t const o_p2 = tabs.add(cv2);
        std::size_t const o_pd = tabs.add(cvd);
        chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
        if(arena.begin(stream))
        {
            void const *table_base = arena.upload_tables(tabs);
            if(table_base != nullptr)
            {
                kernel::hypot::chunked<T>(stream, args->nelems,
                    args->alpha,
                    arena.table_view<const T *>(table_base, o_p1),
                    Index(cv1.nchunks),
                    args->beta,
                    arena.table_view<const T *>(table_base, o_p2),
                    Index(cv2.nchunks),
                    arena.table_view<T *>(table_base, o_pd),
                    Index(cvd.nchunks),
                    Index(cv1.chunk_elems), Index(cv2.chunk_elems),
                    Index(cvd.chunk_elems));
                arena.end(stream);
                return;
            }
            std::fprintf(stderr,
                "nntile_hypot: chunk-ptr table cannot serve this launch\n");
            std::abort();
        }
    }
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for hypot tasks that depends only on cl_arg
template<typename T>
std::uint64_t Hypot<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot task
template<typename T>
void Hypot<std::tuple<T>>::submit(int worker_hint,
        Index nelems, Scalar alpha, ::nnhaul::Handle & src1, Scalar beta, ::nnhaul::Handle & src2, ::nnhaul::Handle & dst)
//! Insert hypot task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_R, &src1 }, { ACCESS_R, &src2 }, { ACCESS_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Hypot<std::tuple<nntile::fp64_t>>;
template class Hypot<std::tuple<nntile::fp32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Hypot<std::tuple<nntile::bf16_t>>;
template class Hypot<std::tuple<nntile::fp16_t>>;

//! Pack of hypot operations for different types
hypot_pack_t hypot;

} // namespace nntile::nnhaul
