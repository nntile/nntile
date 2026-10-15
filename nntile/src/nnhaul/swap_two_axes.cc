/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/starpu/swap_two_axes.cc
 * swap_two_axes operation for StarPU buffers.
 *
 * @version 1.1.0
 * */

#include "nntile/nnhaul/ops/swap_two_axes.hh"
#include "nntile/nnhaul/stage.hh"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "nntile/kernel/swap_two_axes.hh"

namespace nntile::haul
{

// ---------------------------------------------------------------------------
// DELIBERATELY STAGED (family-2 follow-up C: the in-tree derivation).
//
// Kernel index map (kernel::swap_two_axes::cpu/cuda): element
// (i0, i1, i2, i3, i4) sits at src flat i0*d1*d2*d3*d4 + i1*d2*d3*d4 +
// i2*d3*d4 + i3*d4 + i4 and at dst flat i0*d3*d2*d1*d4 + i3*d2*d1*d4 +
// i2*d1*d4 + i1*d4 + i4 - axes 1 and 3 swap, and d4 is the CONTIGUOUS
// axis on BOTH sides. The two candidate decompositions each fail one
// half of the chunked-tile contract:
//  * the natural chunk-local unit is the d4 RUN (contiguous in both
//    src and dst), but the existing kernel signature has no
//    sub-block/panel form - it always swaps the FULL axes 1 and 3 of
//    the view it is handed - so one d4-run per sub-call is not
//    expressible without a new kernel, and per-element staging of
//    d0*d1*d2*d3 runs (one launch each) is the degenerate regime the
//    transpose guard exists to avoid;
//  * the kernel-EXPRESSIBLE sub-view is the i0 SLAB: a sub-call over
//    i0 in [a, b) with dims 1..4 full is (d0' = b-a, d1, d2, d3, d4)
//    at src + a*d1*d2*d3*d4 / dst + a*d3*d2*d1*d4, because d1*d2*d3 ==
//    d3*d2*d1 makes the in-slab swap equal the global swap. But an i0
//    slab spans all of dims 1..4 - it is chunk-local only when the
//    whole slab (nearly the whole tensor) fits one chunk, which is the
//    unchunked case. Straddling slabs would need whole-slab staging
//    through a tensor-sized temp, i.e. staging with extra steps.
// Hence the universal staging fallback: one gather + the classic kernel
// + one scatter, exact for every geometry. A future panel-style kernel
// variant (per-fiber base pointers along the lines of
// kernel::transpose::panel) would unlock the d4-run decomposition.
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
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    kernel::swap_two_axes::cpu<T>(
        args->d0,
        args->d1,
        args->d2,
        args->d3,
        args->d4,
        src,
        dst);
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
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    kernel::swap_two_axes::cuda<T>(
        stream,
        args->d0,
        args->d1,
        args->d2,
        args->d3,
        args->d4,
        src,
        dst);
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
    const double nflops = static_cast<double>(sizeof(T)) * 2.0 *
        static_cast<double>(d0) * static_cast<double>(d1) *
        static_cast<double>(d2) * static_cast<double>(d3) *
        static_cast<double>(d4);
    ::nntile::haul::stage::staged_insert_task(codelet.raw, starpu_worker_hint,
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
