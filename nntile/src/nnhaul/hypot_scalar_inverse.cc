/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/hypot_scalar_inverse.cc
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.2.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/hypot_scalar_inverse.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_scalar_inverse.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::nnhaul
{

//! Constructor
template<typename T>
HypotScalarInverse<std::tuple<T>>::HypotScalarInverse():
    codelet(
        "nntile_hypot_scalar_inverse",
        &HypotScalarInverse<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &HypotScalarInverse<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &HypotScalarInverse<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot_scalar_inverse operation for NNHaul buffers in CPU
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *d) {
            kernel::hypot_scalar_inverse::cpu<T>(
                n, args->eps, args->alpha, d);
        },
        dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotScalarInverse<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot_scalar_inverse for NNHaul buffers on CUDA
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    chunkunits::require_kernel_index_envelope(
        static_cast<std::size_t>(args->nelems),
        "nntile_hypot_scalar_inverse");
    // Get interfaces
    auto dst = ::nntile::nnhaul::buf_chunks<T>(buffers, 0);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    {
        auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[0]);
        bool const all_plain =
            buf_dst->chunk_count() == 1;
        if(!(all_plain) && args->nelems > 0)
        {
            // Chunked single-launch path: the whole multi-chunk range is
            // issued as ONE chunk-aware kernel reading the shared device
            // chunk-pointer table; the per-element math matches cuda().
            chunkunits::ChunkView<T> const cvd(buf_dst);
            chunkunits::LaunchTables &tabs = chunkunits::launch_tables();
            tabs.clear();
            std::size_t const o_pd = tabs.add(cvd);
            chunkunits::ChunkPtrTable &arena = chunkunits::chunk_ptr_table();
            if(arena.begin(stream))
            {
                void const *table_base = arena.upload_tables(tabs);
                if(table_base != nullptr)
                {
                    T *const *td = arena.table_view<T *>(table_base, o_pd);
                    kernel::hypot_scalar_inverse::chunked<T>(
                        stream,
                        args->nelems,
                        args->eps,
                        args->alpha,
                        td,
                        Index(cvd.nchunks),
                        Index(cvd.chunk_elems));
                    arena.end(stream);
                    return;
                }
                    // There is no per-piece
                    // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot table
                    // serve any launch up to a 16 TiB operand, so a failed upload is an
                    // invariant violation, not a mode to stage around.
                    std::fprintf(stderr,
                        "nntile_hypot_scalar_inverse: chunk-ptr table cannot serve this launch (invariant "
                        "violation; no per-piece staging fallback exists)\n");
                    std::abort();
            }
        }

    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::nnhaul::for_each_flat_piece(
        args->nelems,
        [&](std::size_t n, T *d) {
            kernel::hypot_scalar_inverse::cuda<T>(
                stream, n, args->eps, args->alpha, d);
        },
        dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void HypotScalarInverse<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
std::uint64_t HypotScalarInverse<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::nnhaul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot_scalar_inverse task
template<typename T>
void HypotScalarInverse<std::tuple<T>>::submit(int worker_hint,
        Index nelems, Scalar eps, Scalar alpha, ::nnhaul::Handle & dst)
//! Insert hypot_scalar_inverse task into NNHaul task pool
/*! No argument checking is performed. All the inputs are packed and passed to
 * submit() call. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->eps = eps;
    args->alpha = alpha;
    // Submit task
    ::nntile::nnhaul::insert_task(codelet.raw, worker_hint,
        { { ACCESS_RW, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class HypotScalarInverse<std::tuple<nntile::fp64_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_tf32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_fp16_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_bf16_t>>;
template class HypotScalarInverse<std::tuple<nntile::bf16_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp16_t>>;

//! Pack of hypot_scalar_inverse operations for different types
hypot_scalar_inverse_pack_t hypot_scalar_inverse;

} // namespace nntile::nnhaul
