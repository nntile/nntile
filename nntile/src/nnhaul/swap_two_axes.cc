/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/swap_two_axes.cc
 * swap_two_axes operation for StarPU buffers.
 *
 * @version 1.1.0
 * */

#include "nntile/nnhaul/ops/swap_two_axes.hh"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "nntile/kernel/swap_two_axes.hh"
#include "nntile/kernel/nnhaul/chunk_ops.hh"
#include "nntile/nnhaul/chunk_units.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// Native chunked driver (no staging).
//
// Kernel index map (kernel::swap_two_axes::cpu/cuda): element
// (i0, i1, i2, i3, i4) sits at src flat i0*d1*d2*d3*d4 + i1*d2*d3*d4 +
// i2*d3*d4 + i3*d4 + i4 and at dst flat i0*d3*d2*d1*d4 + i3*d2*d1*d4 +
// i2*d1*d4 + i1*d4 + i4 - axes 1 and 3 swap, and d4 is the CONTIGUOUS
// axis on BOTH sides. The chunked body calls the device kernel that
// reads a destination flat position, decodes it with the destination
// axes and writes the corresponding source position; both positions go
// through the chunk tables with the compile-time shift/mask map. A
// chunked buffer is therefore exactly the unchunked call plus pointer
// arithmetic. All-unchunked buffers keep the single classic call.
// ---------------------------------------------------------------------------

template<typename T>
SwapTwoAxes<std::tuple<T>>::SwapTwoAxes():
    codelet(
        "nntile_swap_two_axes",
        &SwapTwoAxes<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SwapTwoAxes<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SwapTwoAxes<std::tuple<T>>::footprint)
{
}

template<typename T>
void SwapTwoAxes<std::tuple<T>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
    {
        kernel::swap_two_axes::cpu<T>(
            args->d0,
            args->d1,
            args->d2,
            args->d3,
            args->d4,
            buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    // Chunked CPU body (never scheduled by this runtime): the same map
    // as the device kernel, elementwise through the chunk views.
    std::size_t const d0 = static_cast<std::size_t>(args->d0);
    std::size_t const d1 = static_cast<std::size_t>(args->d1);
    std::size_t const d2 = static_cast<std::size_t>(args->d2);
    std::size_t const d3 = static_cast<std::size_t>(args->d3);
    std::size_t const d4 = static_cast<std::size_t>(args->d4);
    chunkunits::ChunkView<T> const cs(buf_src, 0);
    chunkunits::ChunkView<T> const cd(buf_dst, 0);
    std::size_t const s0 = d3 * d2 * d1 * d4;
    std::size_t const s3 = d2 * d1 * d4;
    std::size_t const s2 = d1 * d4;
    for(std::size_t p = 0; p < d0 * d1 * d2 * d3 * d4; ++p)
    {
        std::size_t r = p;
        std::size_t const i0 = r / s0;
        r -= i0 * s0;
        std::size_t const i3 = r / s3;
        r -= i3 * s3;
        std::size_t const i2 = r / s2;
        r -= i2 * s2;
        std::size_t const i1 = r / d4;
        std::size_t const i4 = r - i1 * d4;
        std::size_t const sf =
            ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4) + i4;
        std::size_t const sc = cs.chunk_of(sf);
        std::size_t const dc = cd.chunk_of(p);
        *cd.ptr(dc, cd.chunk_off(p)) = *cs.ptr(sc, cs.chunk_off(sf));
    }
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_tf32_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_fp16_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_bf16_t>>::cpu(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


#ifdef NNTILE_USE_CUDA
template<typename T>
void SwapTwoAxes<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    auto buf_src = static_cast<::nnhaul::Buffer *>(buffers[0]);
    auto buf_dst = static_cast<::nnhaul::Buffer *>(buffers[1]);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    if(buf_src->chunk_count() == 1 && buf_dst->chunk_count() == 1)
    {
        // Unchunked fast path: the historical single kernel call, bitwise.
        kernel::swap_two_axes::cuda<T>(
            stream,
            args->d0,
            args->d1,
            args->d2,
            args->d3,
            args->d4,
            buf_src->get_ptr<T>(),
            buf_dst->get_ptr<T>());
        return;
    }
    ::nntile::kernel::nnhaul::chunk_swap_two_axes(
        *buf_src,
        *buf_dst,
        static_cast<std::size_t>(args->d0),
        static_cast<std::size_t>(args->d1),
        static_cast<std::size_t>(args->d2),
        static_cast<std::size_t>(args->d3),
        static_cast<std::size_t>(args->d4),
        chunkunits::nnhaul_elem_type<T>());
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SwapTwoAxes<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args) noexcept
{
    SwapTwoAxes<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
std::uint64_t SwapTwoAxes<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->d0, sizeof(args->d0), hash);
    hash = ::nntile::haul::fnv1a(&args->d1, sizeof(args->d1), hash);
    hash = ::nntile::haul::fnv1a(&args->d2, sizeof(args->d2), hash);
    hash = ::nntile::haul::fnv1a(&args->d3, sizeof(args->d3), hash);
    hash = ::nntile::haul::fnv1a(&args->d4, sizeof(args->d4), hash);
    return hash;
}

template<typename T>
void SwapTwoAxes<std::tuple<T>>::submit(
    int starpu_worker_hint,
    Index d0,
    Index d1,
    Index d2,
    Index d3,
    Index d4,
    ::nnhaul::Handle & src,
    ::nnhaul::Handle & dst)
{
    args_t *args = static_cast<args_t *>(std::malloc(sizeof(*args)));
    args->d0 = d0;
    args->d1 = d1;
    args->d2 = d2;
    args->d3 = d3;
    args->d4 = d4;
    (void)(static_cast<double>(sizeof(T)) * 2.0 *
        static_cast<double>(d0) * static_cast<double>(d1) *
        static_cast<double>(d2) * static_cast<double>(d3) *
        static_cast<double>(d4));
    // The body picks the unchunked classic call or the native chunked
    // kernel; no staging either way.
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

template class SwapTwoAxes<std::tuple<nntile::fp64_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SwapTwoAxes<std::tuple<nntile::bf16_t>>;
template class SwapTwoAxes<std::tuple<nntile::fp16_t>>;

swap_two_axes_pack_t swap_two_axes;

} // namespace nntile::haul
