/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file include/nntile/nnhaul/ops/sqrt_inplace.hh
 * Inplace sqrt operation on a NNHaul buffer
 *
 * @version 1.1.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard libraries
#include <tuple>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::nnhaul
{

//! Generic wrapper class for sqrt_inplace operation is not defined
template<typename T>
class SqrtInplace;

//! Specialization of wrapper class for sqrt_inplace operation via std::tuple
template<typename T>
class SqrtInplace<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    SqrtInplace();

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


#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    //! Submit sqrt_inplace task
    void submit(
        int worker_hint,
        Index nelems,
        ::nnhaul::Handle & data
    );
};

//! Pack of sqrt_inplace operations for different types
using sqrt_inplace_pack_t = OperationPack<
    SqrtInplace,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of sqrt_inplace operations for different types
extern sqrt_inplace_pack_t sqrt_inplace;

} // namespace nntile::nnhaul
