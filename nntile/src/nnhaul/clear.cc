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
    codelet("nntile_clear", &Clear::cpu, nullptr, nullptr)
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
    // Get interfaces
    std::size_t nbytes = (*reinterpret_cast<std::size_t const *>(cl_args));
    void *data = ::nntile::haul::buf_as<void>(buffers, 0);
    // Clear buffer
    std::memset(data, 0, nbytes);
}


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
