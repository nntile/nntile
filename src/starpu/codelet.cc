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
    std::initializer_list<starpu_cpu_func_t> cpu_funcs,
    std::initializer_list<starpu_cuda_func_t> cuda_funcs
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

    // Set runtime decision on number of buffers and access modes
    starpu_codelet::nbuffers = STARPU_VARIABLE_NBUFFERS;

    // Check if the number of CPU implementations is too large
    if(cpu_funcs.size() > STARPU_MAXIMPLEMENTATIONS)
    {
        throw std::runtime_error("Too many CPU func implementations");
    }

    // Add CPU implementations
    if(cpu_funcs.size() > 0)
    {
        auto it = cpu_funcs.begin();
        for(int i = 0; i < cpu_funcs.size(); ++i, ++it)
        {
            if(*it)
            {
//#ifdef STARPU_SIMGRID // Put fake function address in case of simulation
//                    starpu_codelet::cpu_funcs[i] = (starpu_cpu_func_t)0;
//#else // Put real function address
                starpu_codelet::cpu_funcs[i] = *it;
//#endif
                starpu_codelet::where = where_default = STARPU_CPU;
            }
        }
    }

#ifdef NNTILE_USE_CUDA
    // Check if the number of CUDA implementations is too large
    if(cuda_funcs.size() > STARPU_MAXIMPLEMENTATIONS)
    {
        throw std::runtime_error("Too many CUDA func implementations");
    }

    // Add CUDA implementations
    if(cuda_funcs.size() > 0)
    {
        auto it = cuda_funcs.begin();
        for(int i = 0; i < cuda_funcs.size(); ++i, ++it)
        {
            if(*it)
            {
//#ifdef STARPU_SIMGRID // Put fake function address in case of simulation
//                    starpu_codelet::cuda_funcs[i] = (starpu_cuda_func_t)0;
//#else // Put real function address
                starpu_codelet::cuda_funcs[i] = *it;
//#endif
                starpu_codelet::cuda_flags[i] = STARPU_CUDA_ASYNC;
                where_default = where_default | STARPU_CUDA;
                starpu_codelet::where = where_default;
            }
        }
    }
#endif // NNTILE_USE_CUDA
}

//! Restrict where the codelet should be executed
void Codelet::restrict_where(uint32_t where)
{
    // Restrict only if the provided where is supported
    if((where_default & where) == where)
    {
        starpu_codelet::where = where;
    }
}

//! Restore where the codelet should be executed
void Codelet::restore_where()
{
    // Restore the default where
    starpu_codelet::where = where_default;
}

} // namespace nntile
