/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/adam_step.hh
 * Adam step with StarPU buffers
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

//! Generic wrapper class for adam_step operation is not defined
template<typename T>
class AdamStep;

//! Specialization of wrapper class for adam_step operation via std::tuple
template<typename T>
class AdamStep<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    AdamStep();

    //! Structure for operation arguments
    struct args_t
    {
        Index num_iter;
        Index num_elems;
        Scalar beta_1;
        Scalar beta_2;
        Scalar eps;
        Scalar lr;
        Scalar weight_decay;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit adam_step task
    void submit(
        int starpu_worker_hint,
        Index num_iter,
        Index num_elems,
        Scalar beta_1,
        Scalar beta_2,
        Scalar eps,
        Scalar lr,
        Scalar weight_decay,
        ::nnhaul::Handle & grad,
        ::nnhaul::Handle & first_moment,
        ::nnhaul::Handle & second_moment,
        ::nnhaul::Handle & param
    );
};

//! Pack of adam_step operations for different types
using adam_step_pack_t = OperationPack<
    AdamStep,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of adam_step operations for different types
extern adam_step_pack_t adam_step;

} // namespace nntile::haul
