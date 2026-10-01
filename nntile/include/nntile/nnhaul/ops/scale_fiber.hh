/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/scale_fiber.hh
 * StarPU wrappers for scaling of a tensor with a broadcasted fiber
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard headers
#include <tuple>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

//! Generic wrapper class for scale_fiber operation is not defined
template<typename T>
class ScaleFiber;

//! Specialization of wrapper class for scale_fiber operation
template<typename T>
class ScaleFiber<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    ScaleFiber();

    //! Structure for operation arguments
    struct args_t
    {
        Index m;
        Index n;
        Index k;
        Index batch;
        Scalar alpha;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit scale task
    void submit(
        int starpu_worker_hint,
        Index m,
        Index n,
        Index k,
        Index batch,
        Scalar alpha,
        ::nnhaul::Handle & src,
        ::nnhaul::Handle & dst
    );
};

//! Pack of scale_fiber operations for different types
using scale_fiber_pack_t = OperationPack<
    ScaleFiber,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of scale_fiber operations for different types
extern scale_fiber_pack_t scale_fiber;

} // namespace nntile::haul
