/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/norm_fiber_inplace.hh
 * Euclidean norms over slices into a fiber of a product of a StarPU buffer
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

//! Generic wrapper class for norm_fiber_inplace operation is not defined
template<typename T>
class NormFiberInplace;

//! Specialization of wrapper class for norm_fiber_inplace operation via std::tuple
template<typename T>
class NormFiberInplace<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    NormFiberInplace();

    //! Structure for operation arguments
    struct args_t
    {
        Index m;
        Index n;
        Index k;
        Index batch;
        Scalar alpha;
        Scalar beta;
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

    //! Submit norm_fiber_inplace task
    void submit(
        Index m,
        Index n,
        Index k,
        Index batch,
        Scalar alpha,
        Handle src,
        Scalar beta,
        Handle dst,
        int redux=0
    );
};

//! Pack of norm_fiber_inplace operations for different types
using norm_fiber_inplace_pack_t = OperationPack<
    NormFiberInplace,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of norm_fiber_inplace operations for different types
extern norm_fiber_inplace_pack_t norm_fiber_inplace;

} // namespace nntile::starpu
