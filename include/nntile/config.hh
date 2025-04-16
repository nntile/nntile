/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/config.hh
 * Base configuration of NNTile with its initialization
 *
 * @version 1.1.0
 * */

#pragma once

namespace nntile
{

//! Convenient NNTile configuration class
class Config
{
    //! Whether the NNTile and StarPU are initialized
    bool initialized = false;
    //! Whether logger is enabled with the configuration
    bool logger = false;
    //! Logger server address
    const char *logger_server_addr = "localhost";
    //! Logger server port
    int logger_server_port = 5001;
    //! Verbosity level
    int verbose = 0;
public:
    // Constructor is the default one
    Config() = default;

    //! Proper destructor for the only available configuration object
    ~Config()
    {
        // Nothing happens if it was not initialized or already shut down
        shutdown();
    }

    //! Initialize StarPU and NNTile with the configuration
    void init(
        int ncpus=-1,
        int ncuda=-1,
        int cublas=1,
        int ooc=0,
        const char *ooc_path="/tmp/nntile_ooc",
        size_t ooc_size=16777216,
        int ooc_disk_node_id=-1,
        int logger=1,
        const char *logger_server_addr="localhost",
        int logger_server_port=5001,
        int verbose=0
    );

    void shutdown();
};

//! Global NNTile configuration object
extern Config config;

} // namespace nntile
