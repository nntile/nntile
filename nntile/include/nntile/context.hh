/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/context.hh
 * NNTile context
 *
 * @version 2.0.0
 * */

#pragma once

#include <cstddef>

// NNTile definitions
#include <nntile/defs.h>

// Third-party headers
#ifdef NNTILE_USE_CUDA
#endif // NNTILE_USE_CUDA

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

    //! Verbosity level
    int verbose;

    // //! Default value of tiling size along each named dimension
    // std::unordered_map<std::string, Index> default_tile_size;

    //! Constructor of the context
    /*! The only way to initialize the context is to call this constructor.
    * @param[in] ncpu: accepted for compatibility. NNHaul creates no CPU
    *      workers, so this value is ignored.
    * @param[in] ncuda: only one CUDA worker exists. Values <= 0 select it;
    *      values > 1 are rejected. Select the GPU with
    *      ``CUDA_VISIBLE_DEVICES``; ``NNHAUL_DEVICES`` is rejected.
    * @param[in] ooc: must be 0; disk out-of-core is not supported.
    * @param[in] ooc_path: ignored (kept for source compatibility)
    * @param[in] ooc_size: ignored (kept for source compatibility)
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
        int verbose=0,
        std::size_t cpu_cap_bytes=0,
        std::size_t cuda_cap_bytes_each=0
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
