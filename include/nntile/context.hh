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
#include <unordered_set>
#include <mutex>

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

    //! Container for all registered data handles
    std::unordered_set<starpu_data_handle_t> data_handles;

    //! Container for all automatically unregistered data handles
    std::unordered_set<starpu_data_handle_t> data_handles_auto_unreg;

    //! Mutex for a case of multi-threaded garbage collection
    std::mutex data_handles_mutex;

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

    //! Register a data handle into the context
    void data_handle_register(starpu_data_handle_t handle);

    //! Pop a data handle from the container
    bool data_handle_pop(starpu_data_handle_t handle);

    //! Unregister a data handle from the context
    void data_handle_unregister(starpu_data_handle_t handle);

    //! Unregister a data handle from the context without coherency
    void data_handle_unregister_no_coherency(starpu_data_handle_t handle);

    //! Unregister a data handle from the context in an async manner
    void data_handle_unregister_submit(starpu_data_handle_t handle);

    //! Unregister all data handles
    /* This procedure moves all unregistered data handles into a separate
     * container to double deregistration of the same data handles due to
     * order of objects destruction, which is guaranteed to be reverse to the
     * order of objects creation for the C++ language, but not for the Python
     * language. */
    void data_handle_unregister_all();
};

} // namespace nntile
