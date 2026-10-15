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
#include <nntile/nnhaul/codelet.hh>
#include <nntile/constants.hh>

#include <atomic>
#include <cstdint>

namespace nntile::haul
{

//! Number of gemm bodies that ran the native chunked driver so far
//! (the engagement probe for the chunked-tile tests: gemm must NOT go
//! through the staging fallback).
inline std::atomic<std::uint64_t> g_gemm_native_chunked_count{0};

inline void note_gemm_native_chunked() noexcept
{
    g_gemm_native_chunked_count.fetch_add(1, std::memory_order_relaxed);
}

//! How many gemm bodies took the native chunked path so far.
inline std::uint64_t gemm_native_chunked_count() noexcept
{
    return g_gemm_native_chunked_count.load(std::memory_order_relaxed);
}

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
        bool broadcast_a;
        bool broadcast_b;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

#ifdef NNTILE_USE_CBLAS
    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

    //! Array of all wrappers for CPU implementations
#else // NNTILE_USE_CBLAS
    //! Array of all wrappers for CPU implementations
#endif // NNTILE_USE_CBLAS


#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    //! Submit gemm task
    void submit(
        int starpu_worker_hint,
        const TransOp &transA,
        const TransOp &transB,
        Index m,
        Index n,
        Index k,
        Index batch,
        Scalar alpha,
        ::nnhaul::Handle & A,
        ::nnhaul::Handle & B,
        Scalar beta,
        ::nnhaul::Handle & C,
        int redux=0,
        bool broadcast_a=false,
        bool broadcast_b=false
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
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of gemm operations for different types
extern gemm_pack_t gemm;

} // namespace nntile::haul
