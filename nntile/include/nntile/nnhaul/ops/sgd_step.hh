/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/sgd_step.hh
 * SGD with momentum step with StarPU buffers
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

//! Generic wrapper class for sgd_step operation is not defined
template<typename T>
class SGDStep;

//! Specialization of wrapper class for sgd_step operation via std::tuple
template<typename T>
class SGDStep<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    SGDStep();

    //! Structure for operation arguments
    struct args_t
    {
        Index num_iter;
        Index num_elems;
        Scalar momentum;
        Scalar lr;
        Scalar weight_decay;
        Scalar dampening;
        bool nesterov;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit sgd_step task
    void submit(
        int starpu_worker_hint,
        Index num_iter,
        Index num_elems,
        Scalar momentum,
        Scalar lr,
        Scalar weight_decay,
        Scalar dampening,
        bool nesterov,
        ::nnhaul::Handle & grad,
        ::nnhaul::Handle & velocity,
        ::nnhaul::Handle & param
    );
};

//! Pack of sgd_step operations for different types
using sgd_step_pack_t = OperationPack<
    SGDStep,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of sgd_step operations for different types
extern sgd_step_pack_t sgd_step;

} // namespace nntile::haul
