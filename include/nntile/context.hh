/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/context.hh
 * NNTile context
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers

// Third-party headers
#include <starpu.h>

// Other NNTile headers

namespace nntile
{

//! NNTile context
class Context
{
public:
    //! Flag if the context is initialized
    int initialized = 0;

    //! StarPU configuration without explicit default value
    starpu_conf starpu_config;

    //! Whether cuBLAS is enabled
    int cublas;

    //! Whether Out-of-Core is enabled
    int ooc;

    //! Out-of-core disk node id
    int ooc_disk_node_id;

    //! Whether logger is enabled
    int logger;

    //! Verbosity level
    int verbose;

    //! Constructor of the context
    Context(
        int ncpus=-1,
        int ncuda=-1,
        int cublas=1,
        int ooc=0,
        const char *ooc_path="/tmp/nntile_ooc",
        size_t ooc_size=16777216,
        int logger=0,
        const char *logger_server_addr="localhost",
        int logger_server_port=5001,
        int verbose=0
    );

    //! Destructor of the context
    ~Context()
    {
        // It is safe to call shutdown multiple times
        shutdown();
    }

    //! Shut down the context on demand
    void shutdown();
};

} // namespace nntile
