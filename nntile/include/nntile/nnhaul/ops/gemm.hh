/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file include/nntile/nnhaul/ops/gemm.hh
 * GEMM operation for NNHaul buffers
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>
#include <nntile/constants.hh>

#include <cstdint>

namespace nntile::nnhaul
{

//! GEMM staging statistics for the current process.
//!
//! The gemm body is simple: an operand whose chunks are contiguous (or
//! that has only one chunk) is handed to cuBLAS directly; any other
//! operand is copied into the contiguous workspace first. c2w_chunks /
//! w2c_chunks count the chunk-to-workspace and workspace-to-chunk copies
//! the staged path performed (an output C always contributes a W2C copy;
//! C contributes a C2W copy only when beta != 0, because only then is it
//! read). Everything else is diagnostic and never affects control flow.
struct GemmStagingStats
{
    std::uint64_t calls = 0;       //!< every gemm body
    std::uint64_t contiguous = 0;  //!< direct cuBLAS, no copy
    std::uint64_t staged = 0;      //!< workspace-staged path
    std::uint64_t staged_a = 0;    //!< per-operand staging
    std::uint64_t staged_b = 0;
    std::uint64_t staged_c = 0;
    std::uint64_t c2w_chunks = 0;  //!< chunks copied chunked->workspace
    std::uint64_t w2c_chunks = 0;  //!< chunks copied workspace->chunked
    std::uint64_t c2w_bytes = 0;
    std::uint64_t w2c_bytes = 0;
};

//! Snapshot of the process-wide gemm staging statistics.
GemmStagingStats gemm_staging_stats() noexcept;

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

    //! Array of all wrappers for CPU implementations


#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    //! Submit gemm task
    void submit(
        int worker_hint,
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

} // namespace nntile::nnhaul
