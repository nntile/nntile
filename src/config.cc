/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/config.cc
 * Base configuration of NNTile with its initialization
 *
 * @version 1.1.0
 * */

#include "nntile/config.hh"
#include "nntile/defs.h"

namespace nntile
{

//! Initialize entire software stack for the NNTile
void Config::init(int &argc, char **&argv)
{
    // At first initialize StarPU with MPI and cuBLAS support
    int ret;
    ret = starpu_conf_init(this);
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_conf_init()");
    }
    ret = starpu_initialize(&argc, &argv, this);
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_initalize()");
    }
    ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, this);
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_mpi_init_conf()");
    }
#ifdef NNTILE_USE_CUDA
    ret = starpu_cublas_init();
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_cublas_init()");
    }
#endif
    // Initialize all codelets
    bias_init();
}

//! Finalize entire software stack
void Config::shutdown()
{
    // One by one shutdown cuBLAS, MPI and StarPU
    int ret;
#ifdef NNTILE_USE_CUDA
    ret = starpu_cublas_shutdown();
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_cublas_shutdown()");
    }
#endif
    ret = starpu_mpi_shutdown();
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_mpi_shutdown()");
    }
    ret = starpu_shutdown();
    if(ret != STARPU_SUCCESS)
    {
        throw std::runtime_error("Error in starpu_shutdown()");
    }
}

Config config;

} // namespace nntile
