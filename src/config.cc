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

#include <iostream>
#include "nntile/config.hh"
#include "nntile/logger.hh"
#include "nntile/starpu/config.hh"

namespace nntile
{

void Config::init(int ncpu,
        int ncuda,
        int cublas,
        int ooc,
        const char *ooc_path,
        size_t ooc_size,
        int ooc_disk_node_id,
        int logger,
        const char *logger_server_addr,
        int logger_server_port,
        int verbose)
{
    // Set verbose level
    this->verbose = verbose;

    // Initialize StarPU
    starpu::config.init(ncpu, ncuda, cublas, ooc, ooc_path, ooc_size,
            ooc_disk_node_id, verbose);

    // Initialize logger if enabled
    this->logger = logger;
    if(logger)
    {
        this->logger_server_addr = logger_server_addr;
        this->logger_server_port = logger_server_port;
        nntile::logger::logger_init(logger_server_addr, logger_server_port);
        if(verbose > 0)
        {
            std::cout << "Initialized logger\n";
        }
    }

    // Finally, mark the configuration as initialized
    initialized = true;
    if(verbose > 0)
    {
        std::cout << "Finished initialization of NNTile\n";
    }
}

void Config::shutdown()
{
    // Ignore if not initialized
    if(!initialized)
    {
        return;
    }

    // Shutdown logger if enabled
    if(nntile::logger::logger_running)
    {
        nntile::logger::logger_shutdown();
        if(verbose > 0)
        {
            std::cout << "Shutdown logger\n";
        }
    }

    // Shutdown StarPU
    starpu::config.shutdown();

    // Finally, mark the configuration as uninitialized
    initialized = false;
    if(verbose > 0)
    {
        std::cout << "Finished shutdown of NNTile\n";
    }
}

//! Global NNTile configuration object
Config config;

} // namespace nntile
