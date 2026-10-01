/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/randn.hh
 * Randn operation on StarPU buffer
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

//! Generic wrapper class for randn operation is not defined
template<typename T>
class Randn;

//! Specialization of wrapper class for randn operation via std::tuple
template<typename T>
class Randn<std::tuple<T>>
{
public:
    //! Codelet for the current operation
    CodeletTyped<T> codelet;

    //! Constructor
    Randn();

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

    //! Submit randn task
    void submit(
        int starpu_worker_hint,
        Index ndim,
        Index nelems,
        unsigned long long seed,
        Scalar mean,
        Scalar stddev,
        const std::vector<Index> &start,
        const std::vector<Index> &shape,
        const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape,
        ::nnhaul::Handle & data
    );
};

//! Pack of randn operations for different types
using randn_pack_t = OperationPack<
    Randn,
    std::tuple<nntile::fp64_t>,
    std::tuple<nntile::fp32_t>,
    std::tuple<nntile::fp32_fast_tf32_t>,
    std::tuple<nntile::fp32_fast_fp16_t>,
    std::tuple<nntile::fp32_fast_bf16_t>,
    std::tuple<nntile::bf16_t>
>;

//! Pack of randn operations for different types
extern randn_pack_t randn;

} // namespace nntile::haul
