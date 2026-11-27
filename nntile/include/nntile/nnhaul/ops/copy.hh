/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/copy.hh
 * Copy StarPU buffers
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

//! Wrapper class for copy operation
class Copy
{
public:
    //! Codelet for the current operation
    Codelet codelet;

    //! Constructor
    Copy();

    //! Structure for operation arguments
    struct args_t
    {
        std::size_t nbytes;
    };

    //! Footprint function for the current operation
    static std::uint64_t footprint(void const *cl_args, std::size_t cl_arg_size) noexcept;

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    //! Array of all wrappers for CPU implementations


    //! Submit copy task
    void submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst);
};

//! Copy operation object
extern Copy copy;

} // namespace nntile::haul
