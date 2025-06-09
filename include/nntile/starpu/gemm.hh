/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/gemm.hh
 * GEMM operation for StarPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/starpu/codelet.hh>
#include <nntile/starpu/handle.hh>
#include <nntile/constants.hh>

namespace nntile::starpu
{

//! Generic wrapper class for gemm operation is not defined
template<typename T>
class Gemm;

//! Specialization of wrapper class for gemm operation via std::tuple
template<typename T>
class Gemm<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    Gemm();

    //! Structure for operation arguments
    struct args_t
    {
        TransOp transA;
        TransOp transB;
        Index m;
        Index n;
        Index k;
        Index batch;
        Scalar alpha;
        Scalar beta;
    };

    //! Footprint function for the current operation
    static uint32_t footprint(struct starpu_task *task);

#ifdef NNTILE_USE_CBLAS
    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations
    static constexpr func_array cpu_funcs = {
        cpu
    };
#else // NNTILE_USE_CBLAS
    //! Array of all wrappers for CPU implementations
    static constexpr func_array cpu_funcs = {};
#endif // NNTILE_USE_CBLAS

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

    //! Submit gemm task
    void submit(
        const TransOp &transA,
        const TransOp &transB,
        Index m,
        Index n,
        Index k,
        Index batch,
        Scalar alpha,
        Handle A,
        Handle B,
        Scalar beta,
        Handle C,
        int redux=0
    );
};

//! Pack of gemm operations for different types
using gemm_pack_t = OperationPack<
    Gemm,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of gemm operations for different types
extern gemm_pack_t gemm;

} // namespace nntile::starpu
