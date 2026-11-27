/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/starpu/swap_two_axes.hh
 * swap_two_axes operation on StarPU buffers (5D, swap axes 1 and 3).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#include <tuple>

#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

template<typename T>
class SwapTwoAxes;

template<typename T>
class SwapTwoAxes<std::tuple<T>>
{
public:
    CodeletTyped<T> codelet;

    SwapTwoAxes();

    struct args_t
    {
        Index d0;
        Index d1;
        Index d2;
        Index d3;
        Index d4;
    };

    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    static void cpu(void *buffers[], void *cl_args) noexcept;


#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    void submit(
        int starpu_worker_hint,
        Index d0,
        Index d1,
        Index d2,
        Index d3,
        Index d4,
        ::nnhaul::Handle & src,
        ::nnhaul::Handle & dst);
};

using swap_two_axes_pack_t = OperationPack<
    SwapTwoAxes,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

extern swap_two_axes_pack_t swap_two_axes;

} // namespace nntile::haul
