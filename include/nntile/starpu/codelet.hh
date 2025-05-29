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
#include <array>
#include <vector>

// Third-party headers
#include <starpu.h>

// NNTile headers
#include "nntile/base_types.hh"

namespace nntile::starpu
{

//! Type for array of StarPU wrapper functions
using func_array = std::array<starpu_cpu_func_t, STARPU_MAXIMPLEMENTATIONS>;

//! StarPU codelet+perfmodel wrapper
class Codelet: public starpu_codelet, public starpu_perfmodel
{
public:
    //! Default value for where the codelet should be executed
    /*! Possibles values are (not limited to):
     *  - STARPU_CPU: execute the codelet on CPU only
     *  - STARPU_CUDA: execute the codelet on CUDA only
     *  - STARPU_NOWHERE: do not execute the codelet at all. It requires
     *      codelet to have no CPU or CUDA implementations. A task,
     *      corresponding to such a codelet, is just a synchornization point
     *      and does not require any data to be transferred.
     *  - STARPU_CPU | STARPU_CUDA: execute the codelet on CPU and CUDA
     * */
    uint32_t where_default = STARPU_NOWHERE;

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
        func_array cpu_funcs,
        func_array cuda_funcs
    );

    //! Restrict where the codelet should be executed
    /*! @param[in] where: Where the codelet should be executed, possible values
     *      are STARPU_CPU, STARPU_CUDA, STARPU_NOWHERE and their combinations
     *      of form STARPU_CPU | STARPU_CUDA
     * */
    Codelet &restrict_where(uint32_t where);

    //! Restore where the codelet should be executed
    Codelet &restore_where();

    //! Set modes for the codelet
    /*! @param[in] modes: Modes for the codelet
     * */
    Codelet &set_modes_fixed(std::vector<starpu_data_access_mode> modes);

    //! Set runtime decision on number of buffers and access modes
    /*! This is done by default for all the codelets. */
    Codelet &set_modes_variable();
};

//! Codelet wrapper for a specific type
/*! Wrapping Codelet with additional type information allows easy Codelet
 * lookup procedure to be performed by compiler itself in a compile-time manner.
 * */
template<typename T>
class CodeletTyped: public Codelet
{
public:
    //! Get name of the codelet including type information
    static std::string get_name(const char *base_name)
    {
        return std::string(base_name) + "_" + nntile::type_postfix<T>();
    }

    //! Constructor simply calls the base constructor
    CodeletTyped(
        const char *base_name,
        uint32_t (*footprint)(starpu_task *),
        func_array cpu_funcs,
        func_array cuda_funcs
    ):
        Codelet(
            get_name(base_name).c_str(),
            footprint,
            cpu_funcs,
            cuda_funcs
        )
    {
    }
};

//! Codelet pack for multiple types
/*! CodeletPack is a wrapper for multiple CodeletTyped instances. It allows
 * to create a single object that contains codelets for multiple types at once.
 * */
template<template<typename> typename Functor, typename... Ts>
class CodeletPack: public CodeletTyped<Ts>...
{
public:
    CodeletPack(
        const char *base_name,
        uint32_t (*footprint)(starpu_task *)
    ):
        CodeletTyped<Ts>(
            base_name,
            footprint,
            Functor<Ts>::cpu_funcs,
            Functor<Ts>::cuda_funcs
        )...
    {
    }

    //! Set modes for the codelet pack
    CodeletPack &set_modes_fixed(std::vector<starpu_data_access_mode> modes)
    {
        (static_cast<CodeletTyped<Ts> &>(*this).set_modes_fixed(modes), ...);
        return *this;
    }

    //! Set runtime decision on number of buffers and access modes
    CodeletPack &set_modes_variable()
    {
        (static_cast<CodeletTyped<Ts> &>(*this).set_modes_variable(), ...);
        return *this;
    }

    //! Get codelet for a specific type
    template<typename U>
    CodeletTyped<U> *get_codelet_ptr()
    {
        return static_cast<CodeletTyped<U> *>(this);
    }

    //! Restrict where the codelet pack should be executed
    CodeletPack &restrict_where(uint32_t where)
    {
        (static_cast<CodeletTyped<Ts> &>(*this).restrict_where(where), ...);
        return *this;
    }

    //! Restore where the codelet pack should be executed
    CodeletPack &restore_where()
    {
        (static_cast<CodeletTyped<Ts> &>(*this).restore_where(), ...);
        return *this;
    }
};

} // namespace nntile
