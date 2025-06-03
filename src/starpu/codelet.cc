/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/codelet.cc
 * StarPU codelet wrapper and codelet pack
 *
 * @version 1.1.0
 * */

// Related header
#include "nntile/starpu/codelet.hh"

// NNTile definitions
#include "nntile/defs.h"

// Standard library headers
#include <stdexcept>
#include <cstring>
#include <vector>
#include <mutex>

// Third-party headers

// Other NNTile headers


namespace nntile::starpu
{

//! Default constructor
Codelet::Codelet(
    const char *name,
    uint32_t (*footprint)(starpu_task *),
    func_array cpu_funcs,
    func_array cuda_funcs
)
{
    // Link performance model to the codelet
    starpu_codelet::model = this;

    // Set performance model to history-based
    starpu_perfmodel::type = STARPU_HISTORY_BASED;

    // Set codelet name and performance model symbol
    starpu_codelet::name = name;
    starpu_perfmodel::symbol = name;

    // Set footprint function
    starpu_perfmodel::footprint = footprint;

    // Set runtime decision on number of buffers and access modes by default
    starpu_codelet::nbuffers = STARPU_VARIABLE_NBUFFERS;

    // Add CPU implementations
    for(int i = 0; i < STARPU_MAXIMPLEMENTATIONS; ++i)
    {
        // If the implementation is not nullptr, add it to the codelet
        if(cpu_funcs[i])
        {
            starpu_codelet::cpu_funcs[i] = cpu_funcs[i];
            where_default = where_default | STARPU_CPU;
        }
    }

#ifdef NNTILE_USE_CUDA
    // Add CUDA implementations
    for(int i = 0; i < STARPU_MAXIMPLEMENTATIONS; ++i)
    {
        // If the implementation is not nullptr, add it to the codelet
        if(cuda_funcs[i])
        {
            starpu_codelet::cuda_funcs[i] = cuda_funcs[i];
            where_default = where_default | STARPU_CUDA;
        }
    }
#endif // NNTILE_USE_CUDA

    // Set default value of where
    starpu_codelet::where = where_default;
}

//! Restrict where the codelet should be executed
Codelet &Codelet::restrict_where(uint32_t where)
{
    // Restrict only if the provided where is supported
    if((where_default & where) == where)
    {
        starpu_codelet::where = where;
    }

    // Return the codelet
    return *this;
}

//! Restore where the codelet should be executed
Codelet &Codelet::restore_where()
{
    // Restore the default where
    starpu_codelet::where = where_default;

    // Return the codelet
    return *this;
}

//! Set modes for the codelet
Codelet &Codelet::set_modes_fixed(const std::vector<starpu_data_access_mode> &modes)
{
    // Check if the number of modes is too large
    if(modes.size() > STARPU_NMAXBUFS)
    {
        std::cerr << modes.size() << " " << STARPU_NMAXBUFS << std::endl;
        for(auto mode : modes)
        {
            std::cerr << mode << " ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Too many data access modes");
    }

    // Set number of buffers
    starpu_codelet::nbuffers = modes.size();

    // Set modes
    for(int i = 0; i < starpu_codelet::nbuffers; ++i)
    {
        starpu_codelet::modes[i] = modes[i];
    }

    // Clear all the remaining modes
    for(int i = starpu_codelet::nbuffers; i < STARPU_NMAXBUFS; ++i)
    {
        starpu_codelet::modes[i] = STARPU_NONE;
    }

    // Return the codelet
    return *this;
}

//! Set runtime decision on number of buffers and access modes
Codelet &Codelet::set_modes_variable()
{
    // Indicate that the number of buffers is decided during runtime
    starpu_codelet::nbuffers = STARPU_VARIABLE_NBUFFERS;

    // Clear all the modes
    for(int i = 0; i < STARPU_NMAXBUFS; ++i)
    {
        starpu_codelet::modes[i] = STARPU_NONE;
    }

    // Return the codelet
    return *this;
}

} // namespace nntile
