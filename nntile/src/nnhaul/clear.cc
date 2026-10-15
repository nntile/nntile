/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/clear.hh"

// Standard libraries
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nntile::haul
{

//! Constructor
Clear::Clear():
    codelet(
        "nntile_clear",
        &Clear::cpu,
#ifdef NNTILE_USE_CUDA
        &Clear::cuda,
#else
        nullptr,
#endif
        nullptr)
{
    // Modes cannot be variable for clear operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 1> modes = {
        STARPU_W
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Clear a StarPU buffer on CPU
void Clear::cpu(void *buffers[], void *cl_args)
    noexcept
{
    (void)cl_args;
    // Get interfaces
    auto data = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    // Clear buffer chunk by chunk (single chunk if unchunked)
    for (std::size_t i = 0; i < data.count(); ++i)
    {
        std::memset(data.ptr(i), 0, data.elems(i));
    }
}

#ifdef NNTILE_USE_CUDA
//! Clear an NNHaul buffer on CUDA
void Clear::cuda(void *buffers[], void *cl_args)
    noexcept
{
    (void)cl_args;
    auto data = ::nntile::haul::buf_chunks<unsigned char>(buffers, 0);
    cudaStream_t stream = ::nnhaul::cuda_stream();
    for (std::size_t i = 0; i < data.count(); ++i)
    {
        cudaMemsetAsync(data.ptr(i), 0, data.elems(i), stream);
    }
}
#endif // NNTILE_USE_CUDA

//! Submit clear task
void Clear::submit(int starpu_worker_hint, ::nnhaul::Handle & data)
{
    // Submit task
    std::size_t nnhaul_nbytes = data.nbytes();
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_W, &data } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}

//! Clear operation object
Clear clear;

} // namespace nntile::haul
