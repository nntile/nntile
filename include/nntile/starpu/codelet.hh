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

// Disable STARPU_COMMUTE access mode
#undef STARPU_COMMUTE
#define STARPU_COMMUTE 0

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

    //! Name of the codelet
    /*! This variable is used to store the name of the codelet. It is set in
     * the constructor and can be accessed through the name member variable.
     * It cannot be changed after the constructor is called, as a pointer to
     * raw C string will be still used in StarPU.
     * */
    std::string name_as_string = "nntile_codelet_undefined";

    //! Constructor
    /*! @param[in] name: Name of the codelet, that will be seen in StarPU
     *      graphs and performance modeling files
     *  @param[in] footprint: Footprint function for the codelet
     *  @param[in] cpu_funcs: CPU implementations of the codelet
     *  @param[in] cuda_funcs: CUDA implementations of the codelet
     * */
    Codelet(
        const std::string &name,
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
    template<size_t N>
    Codelet &set_modes_fixed(const std::array<starpu_data_access_mode, N> &modes)
    {
        static_assert(N <= STARPU_NMAXBUFS, "Too many data access modes");

        // Set number of buffers
        starpu_codelet::nbuffers = N;

        // Set modes
        for(int i = 0; i < N; ++i)
        {
            starpu_codelet::modes[i] = modes[i];
        }

        // Clear all the remaining modes
        for(int i = N; i < STARPU_NMAXBUFS; ++i)
        {
            starpu_codelet::modes[i] = STARPU_NONE;
        }

        // Return the codelet
        return *this;
    }

    //! Set runtime decision on number of buffers and access modes
    /*! This is done by default for all the codelets. */
    Codelet &set_modes_variable();
};

//! Codelet wrapper for a specific type
/*! Wrapping Codelet with additional type information allows easy Codelet
 * lookup procedure to be performed by compiler itself in a compile-time manner.
 * */
template<typename... Ts>
class CodeletTyped: public Codelet
{
public:
    //! Get name of the codelet including type information
    static std::string get_name(const std::string &base_name)
    {
        return base_name + "_" + nntile::type_postfix<Ts...>();
    }

    //! Constructor simply calls the base constructor
    CodeletTyped(
        const std::string &base_name,
        uint32_t (*footprint)(starpu_task *),
        func_array cpu_funcs,
        func_array cuda_funcs
    ):
        Codelet(
            get_name(base_name),
            footprint,
            cpu_funcs,
            cuda_funcs
        )
    {
    }
};

//! Pack of operations for different types
template<template<typename> typename Operation, typename... Ts>
class OperationPack: public Operation<Ts>...
{
public:
    OperationPack():
        Operation<Ts>()...
    {
    }

    //! Restrict where the operation pack should be executed
    OperationPack &restrict_where(uint32_t where)
    {
        (static_cast<Operation<Ts> &>(*this).codelet.restrict_where(where), ...);
        return *this;
    }

    //! Restore where the operation pack should be executed
    OperationPack &restore_where()
    {
        (static_cast<Operation<Ts> &>(*this).codelet.restore_where(), ...);
        return *this;
    }

    //! Generic submit function
    template<typename T, typename... Args>
    void submit(Args &&...args)
    {
        static_cast<Operation<T> &>(*this).submit(std::forward<Args>(args)...);
    }
};

} // namespace nntile
