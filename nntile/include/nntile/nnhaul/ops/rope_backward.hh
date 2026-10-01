/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/rope_backward.hh
 * Rotary positional embedding
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

//! StarPU headers
// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

//! Generic wrapper class for rope_backward operation is not defined
template<typename T>
class RopeBackward;

//! Specialization of wrapper class for rope_backward operation via std::tuple
template<typename T>
class RopeBackward<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    RopeBackward();

    //! Structure for operation arguments
    struct args_t
    {
        Index m;
        Index n;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit rope_backward task
    void submit(
        int starpu_worker_hint,
        Index m,
        Index n,
        ::nnhaul::Handle & sin,
        ::nnhaul::Handle & cos,
        ::nnhaul::Handle & dy,
        ::nnhaul::Handle & dx
    );
};

//! Pack of rope_backward operations for different types
using rope_backward_pack_t = OperationPack<
    RopeBackward,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::fp16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of rope_backward operations for different types
extern rope_backward_pack_t rope_backward;

} // namespace nntile::haul
