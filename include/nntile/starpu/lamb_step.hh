/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/lamb_step.hh
 * LAMB step with StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard headers
#include <tuple>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu
{

//! Generic wrapper class for lamb_step operation is not defined
template<typename T>
class LambStep;

//! Specialization of wrapper class for lamb_step operation via std::tuple
template<typename T>
class LambStep<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    LambStep();

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
        Scalar min_trust;
        Scalar max_trust;
    };

    //! Footprint function for the current operation
    static uint32_t footprint(struct starpu_task *task);

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations
    static constexpr func_array cpu_funcs = {
        cpu
    };

#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {
        cuda
    };
#else // NNTILE_USE_CUDA
    //! Array of all wrappers for CUDA implementations
    static constexpr func_array cuda_funcs = {};
#endif // NNTILE_USE_CUDA

    //! Submit lamb_step task
    void submit(
        Index num_iter,
        Index num_elems,
        Scalar beta_1,
        Scalar beta_2,
        Scalar eps,
        Scalar lr,
        Scalar weight_decay,
        Scalar min_trust,
        Scalar max_trust,
        Handle grad,
        Handle first_moment,
        Handle second_moment,
        Handle param
    );
};

//! Pack of lamb_step operations for different types
using lamb_step_pack_t = OperationPack<
    LambStep,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of lamb_step operations for different types
extern lamb_step_pack_t lamb_step;

} // namespace nntile::starpu
