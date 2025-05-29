/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/context.cc
 * NNTile context
 *
 * @version 1.1.0
 * */

// Related header
#include "nntile/context.hh"

// Compile-time NNTile definitions
#include "nntile/defs.h"

// Standard library headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// Third-party headers
#ifdef NNTILE_USE_CUDA
#   include <cudnn_frontend.h>
#endif // NNTILE_USE_CUDA

// Other NNTile headers
#include "nntile/logger.hh"
#include "nntile/starpu/handle.hh"

namespace nntile
{

#ifdef NNTILE_USE_CUDA
//! Global variable for cuDNN handles
static cudnnHandle_t cudnn_handles[STARPU_NMAXWORKERS];

//! Specific function to initialize cuDNN per CUDA worker
static void cudnn_init(void *args [[maybe_unused]])
{
    // Get current worker ID and initialize cuDNN handle
    int worker_id = starpu_worker_get_id();
    cudnnCreate(&cudnn_handles[worker_id]);
    // Get CUDA stream for the current worker and set it to the cuDNN handle
    auto stream = starpu_cuda_get_local_stream();
    cudnnSetStream(cudnn_handles[worker_id], stream);
}

//! Specific function to shut down cuDNN per CUDA worker
static void cudnn_shutdown(void *args [[maybe_unused]])
{
    // Get current worker ID and initialize cuDNN handle
    int worker_id = starpu_worker_get_id();
    cudnnDestroy(cudnn_handles[worker_id]);
}
#endif // NNTILE_USE_CUDA

// Constructor of the singleton context
Context::Context(
    int ncpu,
    int ncuda,
    int ooc,
    const char *ooc_path,
    size_t ooc_size,
    int logger,
    const char *logger_addr,
    int logger_port,
    int verbose
):
    initialized(0),
    ooc_disk_node_id(-1),
    verbose(verbose)
{
    // Throw an error if StarPU is already initialized
    if(starpu_is_initialized())
    {
        throw std::runtime_error("StarPU is already initialized");
    }

    // Override env variable STARPU_WORKERS_NOBIND to disable binding of
    // workers to cores to avoid performance degradation on shared systems
    setenv("STARPU_WORKERS_NOBIND", "1", 1);
    if(verbose > 0)
    {
        std::cout << "Set STARPU_WORKERS_NOBIND to 1\n";
    }

    // Disable automatic start of FXT trace by default
    starpu_fxt_autostart_profiling(0);
    if(verbose > 0)
    {
        std::cout << "Disabled automatic start of FXT trace\n";
    }

    // Init StarPU configuration with default values at first
    int ret = starpu_conf_init(&starpu_config);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_conf_init error");
    }
    if(verbose > 0)
    {
        std::cout << "Initialized StarPU configuration object\n";
    }

    // Unset env variable for number of CPU workers if specified
    if(ncpu != -1)
    {
        if(getenv("STARPU_NCPU") != nullptr)
        {
            unsetenv("STARPU_NCPU");
            if(verbose > 0)
            {
                std::cout << "Unset STARPU_NCPU due to specified number "
                    "of CPU workers in the NNTile configuration\n";
            }
        }
        starpu_config.ncpus = ncpu;
        if(verbose > 0)
        {
            std::cout << "Set STARPU_NCPU to " << ncpu << "\n";
        }
    }

#ifdef NNTILE_USE_CUDA
    // Unset env variable for number of CUDA workers if specified
    if(ncuda != -1)
    {
        if(getenv("STARPU_NCUDA") != nullptr)
        {
            unsetenv("STARPU_NCUDA");
            if(verbose > 0)
            {
                std::cout << "Unset STARPU_NCUDA due to specified number "
                    "of CUDA workers in the NNTile configuration\n";
            }
        }
        starpu_config.ncuda = ncuda;
        if(verbose > 0)
        {
            std::cout << "Set STARPU_NCUDA to " << ncuda << "\n";
        }
    }
#endif // NNTILE_USE_CUDA

    // Set history-based scheduler to utilize performance models in case
    // it was not specified by the user
    if(getenv("STARPU_SCHED") == nullptr)
    {
        starpu_config.sched_policy_name = "dmdasd";
        if(verbose > 0)
        {
            std::cout << "Set STARPU_SCHED to " <<
                starpu_config.sched_policy_name << "\n";
        }
    }

    // Init StarPU now
    ret = starpu_init(&starpu_config);
    if(ret != 0)
    {
        throw std::runtime_error("Error in starpu_initialize()");
    }
    if(verbose > 0)
    {
        // Read number of CPU and CUDA workers from StarPU
        ncpu = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
        ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
        std::cout << "Initialized StarPU with NCPU=" << ncpu <<
            " NCUDA=" << ncuda << "\n";
    }

#ifdef NNTILE_USE_CUDA
    // Initialize cuBLAS
    starpu_cublas_init();
    if(verbose > 0)
    {
        std::cout << "Initialized cuBLAS on all CUDA workers\n";
    }

    // Initialize cuDNN
    starpu_execute_on_each_worker(cudnn_init, nullptr, STARPU_CUDA);
    if(verbose > 0)
    {
        std::cout << "Initialized cuDNN on all CUDA workers\n";
    }
#endif // NNTILE_USE_CUDA

    // Initialize Out-of-Core if enabled
    if(ooc != 0)
    {
        ooc_disk_node_id = starpu_disk_register(
            &starpu_disk_unistd_ops, // Use unistd operations
            reinterpret_cast<void *>(const_cast<char *>(ooc_path)),
            ooc_size
        );
        if(verbose > 0)
        {
            std::cout << "Initialized Out-of-Core disk\n";
        }
    }

    // Initialize logger if enabled
    if(logger != 0)
    {
        logger::logger_init(logger_addr, logger_port);
        if(verbose > 0)
        {
            std::cout << "Initialized logger\n";
        }
    }

    // Finally, tell the user that the context is initialized
    initialized = 1;
    if(verbose > 0)
    {
        std::cout << "NNTile context is initialized\n";
    }
}

//! Shut down the context on demand
void Context::shutdown()
{
    // Check if the context is initialized
    if(initialized == 0)
    {
        return;
    }

    // StarPU must be still initialized
    // Because we only support a single active context for now and the current
    // context is still active
    if(!starpu_is_initialized())
    {
        throw std::runtime_error("StarPU must be still initialized");
    }

    // Shutdown logger if it is running
    if(logger::logger_running)
    {
        logger::logger_shutdown();
        if(verbose > 0)
        {
            std::cout << "Shutdown logger\n";
        }
    }

    // Unregister all remaining data handles
    starpu::data_handle_unregister_all();

#ifdef NNTILE_USE_CUDA
    // Shutdown cuBLAS if enabled
    starpu_cublas_shutdown();
    if(verbose > 0)
    {
        std::cout << "Shutdown cuBLAS on all CUDA workers\n";
    }

    // Shutdown cuDNN if enabled
    starpu_execute_on_each_worker(cudnn_shutdown, nullptr, STARPU_CUDA);
    if(verbose > 0)
    {
        std::cout << "Shutdown cuDNN on all CUDA workers\n";
    }
#endif // NNTILE_USE_CUDA

    // Shutdown StarPU
    starpu_shutdown();
    if(verbose > 0)
    {
        std::cout << "Shutdown StarPU\n";
    };

    // Tell the user that the shutdown is finished
    initialized = 0;
    if(verbose > 0)
    {
        std::cout << "Finished shutdown of NNTile\n";
    }
}

} // namespace nntile
