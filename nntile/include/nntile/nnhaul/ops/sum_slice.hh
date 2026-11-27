/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/sum_slice.hh
 * Sum over fibers into a slice of a StarPU buffer
 *
 * @version 1.1.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// Standard libraries
#include <tuple>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::haul
{

//! Generic wrapper class for sum_slice operation is not defined
template<typename T>
class SumSlice;

//! Specialization of wrapper class for sum_slice operation via std::tuple
template<typename T>
class SumSlice<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    SumSlice();

    //! Structure for operation arguments
    struct args_t
    {
        Index m;
        Index n;
        Index k;
        Scalar alpha;
        Scalar beta;
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

    //! Submit sum_slice task
    void submit(
        int starpu_worker_hint,
        Index m,
        Index n,
        Index k,
        Scalar alpha,
        ::nnhaul::Handle & src,
        Scalar beta,
        ::nnhaul::Handle & dst,
        int redux=0
    );
};

//! Pack of sum_slice operations for different types
using sum_slice_pack_t = OperationPack<
    SumSlice,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>,
    std::tuple<nntile::fp16_t>
>;

//! Pack of sum_slice operations for different types
extern sum_slice_pack_t sum_slice;

} // namespace nntile::haul
