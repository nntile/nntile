/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/multiply_inplace.cc
 * Per-element multiplication of two StarPU buffers
 *
 * @version 1.2.0
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

namespace nntile::haul
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

//! Apply multiply on StarPU buffers on CPU
template<typename T>
void MultiplyInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    auto src = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<T>(buffers, 1);
    // Launch kernel per flat chunk piece (single piece if unchunked)
    ::nntile::haul::for_each_flat_piece(
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
//! Apply multiply on StarPU buffer on CUDA
template<typename T>
void MultiplyInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index nelems = args->nelems;
    Scalar alpha = args->alpha;
    // Get interfaces
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if((buf_src->chunk_count() == 1 && !chunkunits::force_chunked_kernels()) && (buf_dst->chunk_count() == 1 && !chunkunits::force_chunked_kernels())
       )
    {
        // Classic contiguous dispatch: illegal under the force flag (the
        // tripwire proves it never happens in a flag-on run).
        chunkunits::classic_dispatch_tripwire("nntile_multiply_inplace",
            false);
        // Unchunked fast path: the historical single kernel call, bitwise
        kernel::multiply_inplace::cuda<T>(stream, nelems, alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    // Chunked single-launch path: issue the WHOLE flat range as ONE
    // chunk-aware kernel instead of the per-piece driver's one launch per
    // chunk boundary crossing. The chunk base-pointer tables are built on the
    // host and parked in the shared device chunk-ptr ring's device-resident
    // ring (no chunk-count cap); the kernel stages them into shared memory
    // per block. Elementwise and bitwise the whole-buffer call. The per-piece
    // driver below stays as the fallback for a
    // failed table allocation.
    // Piece C (2026-10-18c): the all-1-chunk degenerate launch, missing
    // from this wrapper until now (piece 4's 20-op list and piece 6's
    // extension both skipped it -- the graph shape is 3-chunk so it
    // never fired). Same form as the rest of the flat family: chunk
    // counts read straight off the buffers, before any
    // ChunkView/RingTables work, launch pointer-identical to the
    // classic dispatch (gemm path-F precedent; bitwise by construction,
    // ring untouched).
    if(nelems > 0
        && buf_src->chunk_count() == 1
        && buf_dst->chunk_count() == 1)
    {
        kernel::multiply_inplace::cuda<T>(stream, nelems, alpha,
            buf_src->get_ptr<T>(), buf_dst->get_ptr<T>());
        return;
    }
    chunkunits::ChunkView<T> const cv_src(buf_src);
    chunkunits::ChunkView<T> const cv_dst(buf_dst);
    if(nelems > 0)
    {
        chunkunits::RingTables &tabs = chunkunits::ring_tables();
        tabs.clear();
        std::size_t const o_src_ptrs = tabs.add(cv_src);
        std::size_t const o_dst_ptrs = tabs.add(cv_dst);
        chunkunits::ChunkPtrRing &arena = chunkunits::chunk_ptr_ring();
        if(arena.begin(stream))
        {
            void const *ring_base = arena.upload_tables(tabs);
            if(ring_base != nullptr)
            {
                const T *const *ts = arena.table_view<const T *>(
                    ring_base, o_src_ptrs);
                T *const *td = arena.table_view<T *>(ring_base, o_dst_ptrs);
                kernel::multiply_inplace::chunked<T>(
                    stream,
                    nelems,
                    alpha,
                    ts,
                    Index(cv_src.nchunks),
                    Index(cv_src.chunk_elems),
                    td,
                    Index(cv_dst.nchunks),
                    Index(cv_dst.chunk_elems));
                arena.end(stream);
                return;
            }
                // (rule 11, user directive 2026-10-19) There is no per-piece
                // fallback: the 16 MiB pow2 chunk class and the pow2 2^20-slot ring
                // serve any launch up to a 16 TiB operand, so a failed upload is an
                // invariant violation, not a mode to stage around.
                std::fprintf(stderr,
                    "nntile_multiply_inplace: chunk-ptr ring cannot serve this launch (invariant "
                    "violation; the per-piece staging fallback was deleted by rule "
                    "11)\n");
                std::abort();
        }
    }
    if(nelems > 0)
    {
        chunkunits::classic_dispatch_tripwire("nntile_multiply_inplace",
            false);
    }
    // Launch kernel per flat chunk piece (single piece if unchunked)
    auto src = ::nntile::haul::buf_chunks<T>(buffers, 0);
    auto dst = ::nntile::haul::buf_chunks<T>(buffers, 1);
    ::nntile::haul::for_each_flat_piece(
        nelems,
        [&](std::size_t n, const T *s, T *d) {
            kernel::multiply_inplace::cuda<T>(stream, n, alpha, s, d);
        },
        src,
        dst);
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
    hash = ::nntile::haul::fnv1a(&args->nelems, sizeof(args->nelems), hash);
    hash = ::nntile::haul::fnv1a(&args->alpha, sizeof(args->alpha), hash);
    return hash;
}

//! Submit multiply_inplace operation
template<typename T>
void MultiplyInplace<std::tuple<T>>::submit(int starpu_worker_hint, Index nelems, Scalar alpha, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_RW, &dst } }, args, sizeof(*args));
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

} // namespace nntile::haul
