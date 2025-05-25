/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/codelet.hh
 * StarPU codelet wrapper and codelet pack
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <initializer_list>

// Third-party headers
#include <starpu.h>

// NNTile headers
#include "nntile/base_types.hh"

namespace nntile::starpu
{

//! StarPU codelet+perfmodel wrapper
class Codelet: public starpu_codelet, public starpu_perfmodel
{
public:
    //! Default value for where the codelet should be executed
    uint32_t where_default;

    //! No default constructor
    Codelet() = delete;

    //! Constructor
    /*! @param[in] name: Name of the codelet, that will be seen in StarPU
     *      graphs and performance modeling files
     *  @param[in] footprint: Footprint function for the codelet
     *  @param[in] cpu_funcs: CPU implementations of the codelet
     *  @param[in] cuda_funcs: CUDA implementations of the codelet
     * */
    Codelet(
        const char *name,
        uint32_t (*footprint)(starpu_task *),
        std::initializer_list<starpu_cpu_func_t> cpu_funcs,
        std::initializer_list<starpu_cuda_func_t> cuda_funcs
    );

    //! Restrict where the codelet should be executed
    /*! @param[in] where: Where the codelet should be executed, possible values
     *      are STARPU_CPU, STARPU_CUDA, STARPU_NOWHERE and their combinations
     *      of form STARPU_CPU | STARPU_CUDA
     * */
    void restrict_where(uint32_t where);

    //! Restore where the codelet should be executed
    void restore_where();
};

} // namespace nntile
