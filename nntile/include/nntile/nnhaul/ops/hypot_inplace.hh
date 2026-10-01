/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/hypot_inplace.hh
 * hypot_inplace operation on StarPU buffers
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

//! Generic wrapper class for hypot_inplace operation is not defined
template<typename T>
class HypotInplace;

//! Specialization of wrapper class for hypot_inplace operation via std::tuple
template<typename T>
class HypotInplace<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    HypotInplace();

    //! Structure for operation arguments
    struct args_t
    {
        Index nelems;
        Scalar alpha;
        Scalar beta;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit hypot_inplace task
    void submit(
        int starpu_worker_hint,
        Index nelems,
        Scalar alpha,
        ::nnhaul::Handle & src,
        Scalar beta,
        ::nnhaul::Handle & dst
    );
};

//! Pack of hypot_inplace operations for different types
using hypot_inplace_pack_t = OperationPack<
    HypotInplace,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of hypot_inplace operations for different types
extern hypot_inplace_pack_t hypot_inplace;

} // namespace nntile::haul
