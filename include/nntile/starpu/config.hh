/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/config.hh
 * StarPU configuration, data handles and codelets base classes
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>
#include <vector>
#include <cstring>
#include <iostream>
#include <starpu.h>
// Disabled MPI for now
//#include <starpu_mpi.h>
#include <nntile/defs.h>
#include <nntile/context.hh>

namespace nntile
{

// Fake STARPU functions
#define MPI_COMM_WORLD 0

static int starpu_mpi_world_size()
{
    return 1;
}

static int starpu_mpi_world_rank()
{
    return 0;
}

static int starpu_mpi_wait_for_all(int comm)
{
    return 0;
}

static int starpu_mpi_barrier(int comm)
{
    return 0;
}

namespace starpu
{

// Use Config name for legacy code
namespace Config
{

//! StarPU commute data access mode
constexpr starpu_data_access_mode STARPU_RW_COMMUTE
    = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);

// Unpack with no argument remaining
static void unpack_args_ptr_single_arg(void *cl_args, int nargs)
{
}

// Unpack arguments one by one
template<typename T, typename... Ts>
void unpack_args_ptr_single_arg(void *cl_args, int nargs, const T *&ptr,
        const Ts *&...args)
{
    // Do nothing if there are no remaining arguments
    if(nargs == 0)
    {
        return;
    }
    // The first element is a size of argument
    size_t arg_size = reinterpret_cast<size_t *>(cl_args)[0];
    // Get pointer to the data
    char *char_ptr = reinterpret_cast<char *>(cl_args) + sizeof(size_t);
    ptr = reinterpret_cast<T *>(char_ptr);
    // Move pointer by data size
    cl_args = char_ptr + arg_size;
    // Unpack next argument
    unpack_args_ptr_single_arg(cl_args, nargs-1, args...);
}

// Unpack args by pointers without copying actual data
template<typename... Ts>
void unpack_args_ptr(void *cl_args, const Ts *&...args)
{
    // The first element is a total number of packed arguments
    int nargs = reinterpret_cast<int *>(cl_args)[0];
    cl_args = reinterpret_cast<char *>(cl_args) + sizeof(int);
    // Unpack arguments one by one
    if(nargs > 0)
    {
        unpack_args_ptr_single_arg(cl_args, nargs, args...);
    }
}

} // namespace Config

} // namespace starpu
} // namespace nntile
