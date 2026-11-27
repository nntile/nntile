/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/multiply.hh
 * Per-element product of two StarPU buffers
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

//! Generic wrapper class for multiply operation is not defined
template<typename T>
class Multiply;

//! Specialization of wrapper class for multiply operation via std::tuple
template<typename T>
class Multiply<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    Multiply();

    //! Structure for operation arguments
    struct args_t
    {
        Index nelems;
        Scalar alpha;
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

    //! Submit multiply task
    void submit(
        int starpu_worker_hint,
        Index nelems,
        Scalar alpha,
        ::nnhaul::Handle & src1,
        ::nnhaul::Handle & src2,
        ::nnhaul::Handle & dst
    );
};

//! Pack of multiply operations for different types
using multiply_pack_t = OperationPack<
    Multiply,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of multiply operations for different types
extern multiply_pack_t multiply;

} // namespace nntile::haul
