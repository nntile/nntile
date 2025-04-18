/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/codelet.hh
 * StarPU codelet base class
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <initializer_list>

// Third-party headers
#include <starpu.h>

// Other NNTile headers


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
    Codelet(
        const char *name_,
        uint32_t (*footprint_)(starpu_task *),
        std::initializer_list<starpu_cpu_func_t> cpu_funcs_,
        std::initializer_list<starpu_cuda_func_t> cuda_funcs_
    );

    //! Restrict where the codelet should be executed
    void restrict_where(uint32_t where_);

    //! Restore where the codelet should be executed
    void restore_where();
};

// //! Initialize all codelets
// void init_all_codelets();

// //! Restrict all codelets to a given computational unit
// void restrict_all_codelets(uint32_t where);

// //! Restore all codelets
// void restore_all_codelets();

} // namespace nntile
