/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/log_scalar.hh
 * StarPU wrapper to log scalar value
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard headers
#include <string>
#include <tuple>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>

namespace nntile::starpu
{

//! Generic wrapper class for log_scalar operation is not defined
template<typename T>
class LogScalar;

//! Specialization of wrapper class for log_scalar operation via std::tuple
template<typename T>
class LogScalar<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    LogScalar();

    //! Structure for operation arguments
    struct args_t
    {
        std::string name;
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

    //! No CUDA implementations
    static constexpr func_array cuda_funcs = {};

    //! Submit log_scalar task
    void submit(
        const std::string &name,
        Handle value
    );
};

//! Pack of log_scalar operations for different types
using log_scalar_pack_t = OperationPack<
    LogScalar,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of log_scalar operations for different types
extern log_scalar_pack_t log_scalar;

} // namespace nntile::starpu
