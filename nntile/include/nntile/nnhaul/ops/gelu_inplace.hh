/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/gelu_inplace.hh
 * GeLU inplace operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

//! Generic wrapper class for gelu_inplace operation is not defined
template<typename T>
class GeluInplace;

//! Specialization of wrapper class for gelu_inplace operation via std::tuple
template<typename T>
class GeluInplace<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    GeluInplace();

    //! Structure for operation arguments
    struct args_t
    {
        Index nelems;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations


    //! Submit gelu_inplace task
    void submit(
        int starpu_worker_hint,
        Index nelems,
        ::nnhaul::Handle & data
    );
};

//! Pack of gelu_inplace operations for different types
using gelu_inplace_pack_t = OperationPack<
    GeluInplace,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of gelu_inplace operations for different types
extern gelu_inplace_pack_t gelu_inplace;

} // namespace nntile::haul
