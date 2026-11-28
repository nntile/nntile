/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file include/nntile/nnhaul/ops/clear.hh
 * Clear a NNHaul buffer
 *
 * @version 2.0.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

// NNTile headers
#include <nntile/nnhaul/codelet.hh>

namespace nntile::nnhaul
{

//! Wrapper class for clear operation
class Clear
{
public:
    //! Codelet for the current operation
    Codelet codelet;

    //! Constructor
    Clear();

    //! Structure for operation arguments
    struct args_t
    {
    };

    //! Wrapper for a generic CPU implementation
    static void cpu(void *buffers[], void *cl_args)
        noexcept;

#ifdef NNTILE_USE_CUDA
    //! Wrapper for a generic CUDA implementation
    static void cuda(void *buffers[], void *cl_args)
        noexcept;
#endif // NNTILE_USE_CUDA

    //! Array of all wrappers for CPU implementations


    //! Submit clear task
    void submit(int worker_hint, ::nnhaul::Handle & data);
};

//! Clear operation object
extern Clear clear;

} // namespace nntile::nnhaul
