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
/*! It is a singleton that contains all the configuration for the NNTile
 * runtime. It is implemented via a singleton to ensure that entire context
 * is properly initialized before any other NNTile code is executed and that
 * entire context is properly destroyed on exit or error.
 */
class Context
{
public:
    //! Flag if the context is initialized
    int initialized = 0;

    //! StarPU configuration without explicit default value
    starpu_conf starpu_config;

    //! StarPU memory node id of a disk for OOC
    int ooc_disk_node_id;

    //! Verbosity level
    int verbose;

    // //! Default value of tiling size along each named dimension
    // std::unordered_map<std::string, Index> default_tile_size;

    //! Constructor of the context
    /*! The only way to initialize the context is to call this constructor.
    * @param[in] ncpu: number of CPU workers, -1 means that STARPU_NCPU env
    *      variable is used
    * @param[in] ncuda: number of CUDA workers, -1 means that STARPU_NCUDA env
    *      variable is used
    * @param[in] ooc: enable Out-of-Core (OOC) support
    * @param[in] ooc_path: path to the OOC disk
    * @param[in] ooc_size: size of the OOC disk in bytes
    * @param[in] logger: whether logger is enabled
    * @param[in] logger_addr: address of the logger server
    * @param[in] logger_port: port of the logger server
    * @param[in] verbose: whether verbose output is enabled
    */
    Context(
        int ncpu=-1,
        int ncuda=-1,
        int ooc=0,
        const char *ooc_path="/tmp/nntile_ooc",
        size_t ooc_size=16777216,
        int logger=0,
        const char *logger_addr="localhost",
        int logger_port=5001,
        int verbose=0
    );

    //! Destructor of the context
    ~Context()
    {
        // It is safe to call shutdown multiple times
        shutdown();
    }

    //! Shut down the context on demand
    /*! It is safe to call this function multiple times, provided previous
     * calls did not fail.
     */
    void shutdown();
};

} // namespace nntile
