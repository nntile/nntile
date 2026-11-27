/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/accumulate_hypot.hh
 * Accumulate one StarPU buffers into another as hypot
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

//! Generic wrapper class for accumulate_hypot operation is not defined
template<typename T>
class AccumulateHypot;

//! Specialization of wrapper class for add operation via std::tuple
template<typename T>
class AccumulateHypot<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    AccumulateHypot();

    //! Structure for operation arguments
    struct args_t
    {
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

    //! Submit accumulate_hypot task
    void submit(
        int starpu_worker_hint,
        ::nnhaul::Handle & src,
        ::nnhaul::Handle & dst
    );
};

//! Pack of accumulate_hypot operations for different types
using accumulate_hypot_pack_t = OperationPack<
    AccumulateHypot,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of accumulate_hypot operations for different types
extern accumulate_hypot_pack_t accumulate_hypot;

} // namespace nntile::haul
