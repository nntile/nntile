/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/config.cc
 * Base configuration of StarPU with its initialization
 *
 * @version 1.1.0
 * */

#include <iostream>
#include <sstream>
#include <unordered_set>
#include "nntile/starpu/config.hh"
#include "nntile/starpu.hh"

namespace nntile::starpu
{

void Config::init(int ncpus,
        int ncuda,
        int cublas,
        int ooc,
        const char *ooc_path,
        size_t ooc_size,
        int ooc_disk_node_id,
        int verbose)
{
    // Ignore if already initialized
    if(starpu_is_initialized())
    {
        if(verbose > 0)
        {
            std::cout << "StarPU is already initialized, ignoring new init\n";
        }
        return;
    }

    // Set verbose level
    this->verbose = verbose;

    // Init all codelets
    if(!codelets_initialized)
    {
        nntile::starpu::init();
        if(verbose > 0)
        {
            std::cout << "Initialized StarPU codelets\n";
        }
        codelets_initialized = true;
    }

    // Override env variable STARPU_WORKERS_NOBIND
    // to disable binding of workers to cores
    // to avoid performance degradation on shared systems
    setenv("STARPU_WORKERS_NOBIND", "1", 1);
    if(verbose > 0)
    {
        std::cout << "Set STARPU_WORKERS_NOBIND to 1\n";
    }

    // Disable automatic start of profiling by default
    starpu_fxt_autostart_profiling(0);
    if(verbose > 0)
    {
        std::cout << "Disabled automatic start of profiling\n";
    }

    // Init StarPU configuration with default values at first
    int ret = starpu_conf_init(&starpu_config);
    if(ret != 0)
    {
        throw std::runtime_error("starpu_conf_init error");
    }
    if(verbose > 0)
    {
        std::cout << "Initialized StarPU configuration\n";
    }

    // Unset env variable for number of CPU workers if specified
    if(ncpus != -1)
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
        starpu_config.ncpus = ncpus;
        if(verbose > 0)
        {
            std::cout << "Set STARPU_NCPU to " << ncpus << "\n";
        }
    }

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
        int ncpus_ = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
        int ncuda_ = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
        std::cout << "Initialized StarPU with NCPU=" << ncpus_ <<
            " NCUDA=" << ncuda_ << "\n";
    }

    // Initialize cuBLAS if enabled
    this->cublas = cublas;
#ifdef NNTILE_USE_CUDA
    if(cublas)
    {
        starpu_cublas_init();
        if(verbose > 0)
        {
            std::cout << "Initialized cuBLAS\n";
        }
    }
#endif // NNTILE_USE_CUDA

    // Initialize Out-of-Core if enabled
    this->ooc = ooc;
    if(ooc)
    {
        this->ooc_path = ooc_path;
        this->ooc_size = ooc_size;
        this->ooc_disk_node_id = starpu_disk_register(
            &starpu_disk_unistd_ops, // Use unistd operations
            reinterpret_cast<void *>(const_cast<char *>(ooc_path)),
            ooc_size);
        if(verbose > 0)
        {
            std::cout << "Initialized Out-of-Core\n";
        }
    }

    // Tell that StarPU is initialized
    if(verbose > 0)
    {
        std::cout << "Finished initialization of StarPU\n";
    }
}

void Config::shutdown()
{
    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    // Ignore if not initialized
    if(!starpu_is_initialized())
    {
        if(verbose > 0)
        {
            std::cout << "StarPU is not initialized, ignoring shutdown\n";
        }
        return;
    }

    // Unregister all remaining data handles
    if(verbose > 0)
    {
        std::cout << "There are " << data_handles.size()
            << " remaining data handles to unregister\n";
    }
    for(auto handle: data_handles)
    {
        starpu_data_unregister(handle);
        auto_unreg_data_handles.insert(handle);
        if(verbose > 1)
        {
            std::cout << "Automatically unregistered data handle " <<
                handle << "\n";
        }
    }
    auto_unreg_data_handles = std::move(data_handles);
    if(verbose > 0)
    {
        std::cout << "Automatically unregistered all remaining data "
            "handles\n";
    }

    // Shutdown cuBLAS if enabled
#ifdef NNTILE_USE_CUDA
    if(cublas != 0)
    {
        starpu_cublas_shutdown();
        if(verbose > 0)
        {
            std::cout << "Shutdown cuBLAS\n";
        }
    }
#endif // NNTILE_USE_CUDA

    // Shutdown StarPU
    starpu_shutdown();
    if(verbose > 0)
    {
        std::cout << "Shutdown StarPU\n";
    }

    // Set verbose level to zero to avoid printing messages in a case of multiple
    // shutdowns
    verbose = 0;
}

//! Insert a data handle into the container
void Config::data_handle_register(starpu_data_handle_t handle)
{
    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    data_handles.insert(handle);
    if(verbose > 1)
    {
        std::cout << "Registered data handle " << handle << "\n";
    }
}

//! Pop a data handle from the container
bool Config::data_handle_pop(starpu_data_handle_t handle)
{
    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    // Check if the data handle is registered
    auto it = data_handles.find(handle);
    if(it != data_handles.end())
    {
        data_handles.erase(it);
        if(verbose > 1)
        {
            std::cout << "Data handle " << handle << " removed from "
                "registered data handles\n";
        }
        // Tell the caller that the data handle is still registered
        return true;
    }
    // If it is not registered, check if it was automatically unregistered
    // Automatic unregistration is performed in case a user explicitly calls
    // nntile::config.shutdown() or due to Python garbage collector
    auto it_auto = auto_unreg_data_handles.find(handle);
    if(it_auto != auto_unreg_data_handles.end())
    {
        auto_unreg_data_handles.erase(it_auto);
        // Verbose level is set to zero after shutdown, so we do not print
        // Tell the caller that the data handle is not registered
        return false;
    }
    // Throw an error if the data handle is not registered and not
    // automatically unregistered
    std::stringstream ss;
    ss << "Data handle " << handle << " to be unregistered was not found";
    throw std::runtime_error(ss.str());
}

//! Unregister a data handle from the container
void Config::data_handle_unregister(starpu_data_handle_t handle)
{
    // Pop the data handle from the container
    if(data_handle_pop(handle))
    {
        // Unregister the data handle
        starpu_data_unregister(handle);
        if(verbose > 1)
        {
            std::cout << "Unregistered data handle (standard)\n";
        }
    }
}

//! Unregister a data handle from the container without coherency
void Config::data_handle_unregister_no_coherency(starpu_data_handle_t handle)
{
    // Pop the data handle from the container
    if(data_handle_pop(handle))
    {
        // Unregister the data handle without coherency
        starpu_data_unregister_no_coherency(handle);
        if(verbose > 1)
        {
            std::cout << "Unregistered data handle (no coherency)\n";
        }
    }
}

//! Unregister a data handle from the container in an async manner
void Config::data_handle_unregister_submit(starpu_data_handle_t handle)
{
    // Pop the data handle from the container
    if(data_handle_pop(handle))
    {
        // Unregister the data handle in an async manner
        starpu_data_unregister_submit(handle);
        if(verbose > 1)
        {
            std::cout << "Unregistered data handle (submit)\n";
        }
    }
}

//! Global StarPU configuration object
Config config;

} // namespace nntile::starpu
