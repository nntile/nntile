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
#include "nntile/starpu.hh"

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

    // Wait for all tasks to finish
    starpu_task_wait_for_all();

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

//! Restrict computation to CPU
void Context::restrict_cpu()
{
    using namespace starpu;
    accumulate.restrict_where(STARPU_CPU);
    accumulate_hypot.restrict_where(STARPU_CPU);
    accumulate_maxsumexp.restrict_where(STARPU_CPU);
    adam_step.restrict_where(STARPU_CPU);
    adamw_step.restrict_where(STARPU_CPU);
    add.restrict_where(STARPU_CPU);
    add_fiber.restrict_where(STARPU_CPU);
    add_fiber_inplace.restrict_where(STARPU_CPU);
    add_inplace.restrict_where(STARPU_CPU);
    add_slice.restrict_where(STARPU_CPU);
    add_slice_inplace.restrict_where(STARPU_CPU);
    clear.codelet.restrict_where(STARPU_CPU);
    conv2d_bwd_input_inplace.restrict_where(STARPU_CPU);
    conv2d_bwd_weight_inplace.restrict_where(STARPU_CPU);
    conv2d_inplace.restrict_where(STARPU_CPU);
    copy.codelet.restrict_where(STARPU_CPU);
    embedding.restrict_where(STARPU_CPU);
    embedding_backward.restrict_where(STARPU_CPU);
    fill.restrict_where(STARPU_CPU);
    gelu_inplace.restrict_where(STARPU_CPU);
    gelu_backward.restrict_where(STARPU_CPU);
    gelutanh.restrict_where(STARPU_CPU);
    gelutanh_backward.restrict_where(STARPU_CPU);
    gelutanh_inplace.restrict_where(STARPU_CPU);
    gemm.restrict_where(STARPU_CPU);
    hypot_inplace.restrict_where(STARPU_CPU);
    hypot_scalar_inverse.restrict_where(STARPU_CPU);
    log_scalar.restrict_where(STARPU_CPU);
    logsumexp.restrict_where(STARPU_CPU);
    mask_scalar.restrict_where(STARPU_CPU);
    maxsumexp.restrict_where(STARPU_CPU);
    norm_fiber.restrict_where(STARPU_CPU);
    norm_slice_inplace.restrict_where(STARPU_CPU);
    pow.restrict_where(STARPU_CPU);
    prod.restrict_where(STARPU_CPU);
    multiply_fiber_inplace.restrict_where(STARPU_CPU);
    multiply_fiber.restrict_where(STARPU_CPU);
    multiply_inplace.restrict_where(STARPU_CPU);
    multiply_slice.restrict_where(STARPU_CPU);
    randn.restrict_where(STARPU_CPU);
    relu.restrict_where(STARPU_CPU);
    relu_backward.restrict_where(STARPU_CPU);
    relu_forward.restrict_where(STARPU_CPU);
    rope.restrict_where(STARPU_CPU);
    rope_backward.restrict_where(STARPU_CPU);
    scale.restrict_where(STARPU_CPU);
    scale_inplace.restrict_where(STARPU_CPU);
    silu_backward.restrict_where(STARPU_CPU);
    silu_forward.restrict_where(STARPU_CPU);
    softmax.restrict_where(STARPU_CPU);
    softmax_inplace.restrict_where(STARPU_CPU);
    sqrt.restrict_where(STARPU_CPU);
    sqrt_inplace.restrict_where(STARPU_CPU);
    subcopy.restrict_where(STARPU_CPU);
    subtract_indexed_outputs.restrict_where(STARPU_CPU);
    sum_fiber.restrict_where(STARPU_CPU);
    sum_slice.restrict_where(STARPU_CPU);
    sumprod_fiber.restrict_where(STARPU_CPU);
    sumprod_slice.restrict_where(STARPU_CPU);
    total_sum_accum.restrict_where(STARPU_CPU);
    transpose.restrict_where(STARPU_CPU);
}

//! Restrict computation to CUDA
void Context::restrict_cuda()
{
    using namespace starpu;
    accumulate.restrict_where(STARPU_CUDA);
    accumulate_hypot.restrict_where(STARPU_CUDA);
    accumulate_maxsumexp.restrict_where(STARPU_CUDA);
    adam_step.restrict_where(STARPU_CUDA);
    adamw_step.restrict_where(STARPU_CUDA);
    add.restrict_where(STARPU_CUDA);
    add_fiber.restrict_where(STARPU_CUDA);
    add_fiber_inplace.restrict_where(STARPU_CUDA);
    add_inplace.restrict_where(STARPU_CUDA);
    add_slice.restrict_where(STARPU_CUDA);
    add_slice_inplace.restrict_where(STARPU_CUDA);
    clear.codelet.restrict_where(STARPU_CUDA);
    conv2d_bwd_input_inplace.restrict_where(STARPU_CUDA);
    conv2d_bwd_weight_inplace.restrict_where(STARPU_CUDA);
    conv2d_inplace.restrict_where(STARPU_CUDA);
    copy.codelet.restrict_where(STARPU_CUDA);
    embedding.restrict_where(STARPU_CUDA);
    embedding_backward.restrict_where(STARPU_CUDA);
    fill.restrict_where(STARPU_CUDA);
    gelu_inplace.restrict_where(STARPU_CUDA);
    gelu_backward.restrict_where(STARPU_CUDA);
    gelutanh.restrict_where(STARPU_CUDA);
    gelutanh_backward.restrict_where(STARPU_CUDA);
    gelutanh_inplace.restrict_where(STARPU_CUDA);
    gemm.restrict_where(STARPU_CUDA);
    hypot_inplace.restrict_where(STARPU_CUDA);
    hypot_scalar_inverse.restrict_where(STARPU_CUDA);
    log_scalar.restrict_where(STARPU_CUDA);
    logsumexp.restrict_where(STARPU_CUDA);
    mask_scalar.restrict_where(STARPU_CUDA);
    maxsumexp.restrict_where(STARPU_CUDA);
    norm_fiber.restrict_where(STARPU_CUDA);
    norm_slice_inplace.restrict_where(STARPU_CUDA);
    pow.restrict_where(STARPU_CUDA);
    prod.restrict_where(STARPU_CUDA);
    multiply_fiber_inplace.restrict_where(STARPU_CUDA);
    multiply_fiber.restrict_where(STARPU_CUDA);
    multiply_inplace.restrict_where(STARPU_CUDA);
    multiply_slice.restrict_where(STARPU_CUDA);
    randn.restrict_where(STARPU_CUDA);
    relu.restrict_where(STARPU_CUDA);
    relu_backward.restrict_where(STARPU_CUDA);
    relu_forward.restrict_where(STARPU_CUDA);
    rope.restrict_where(STARPU_CUDA);
    rope_backward.restrict_where(STARPU_CUDA);
    scale.restrict_where(STARPU_CUDA);
    scale_inplace.restrict_where(STARPU_CUDA);
    silu_backward.restrict_where(STARPU_CUDA);
    silu_forward.restrict_where(STARPU_CUDA);
    softmax.restrict_where(STARPU_CUDA);
    softmax_inplace.restrict_where(STARPU_CUDA);
    sqrt.restrict_where(STARPU_CUDA);
    sqrt_inplace.restrict_where(STARPU_CUDA);
    subcopy.restrict_where(STARPU_CUDA);
    subtract_indexed_outputs.restrict_where(STARPU_CUDA);
    sum_fiber.restrict_where(STARPU_CUDA);
    sum_slice.restrict_where(STARPU_CUDA);
    sumprod_fiber.restrict_where(STARPU_CUDA);
    sumprod_slice.restrict_where(STARPU_CUDA);
    total_sum_accum.restrict_where(STARPU_CUDA);
    transpose.restrict_where(STARPU_CUDA);
}

//! Restore computation to all devices
void Context::restore_where()
{
    using namespace starpu;
    accumulate.restore_where();
    accumulate_hypot.restore_where();
    accumulate_maxsumexp.restore_where();
    adam_step.restore_where();
    adamw_step.restore_where();
    add.restore_where();
    add_fiber.restore_where();
    add_fiber_inplace.restore_where();
    add_inplace.restore_where();
    add_slice.restore_where();
    add_slice_inplace.restore_where();
    clear.codelet.restore_where();
    conv2d_bwd_input_inplace.restore_where();
    conv2d_bwd_weight_inplace.restore_where();
    conv2d_inplace.restore_where();
    copy.codelet.restore_where();
    embedding.restore_where();
    embedding_backward.restore_where();
    fill.restore_where();
    gelu_inplace.restore_where();
    gelu_backward.restore_where();
    gelutanh.restore_where();
    gelutanh_backward.restore_where();
    gelutanh_inplace.restore_where();
    gemm.restore_where();
    hypot_inplace.restore_where();
    hypot_scalar_inverse.restore_where();
    log_scalar.restore_where();
    logsumexp.restore_where();
    mask_scalar.restore_where();
    maxsumexp.restore_where();
    norm_fiber.restore_where();
    norm_slice_inplace.restore_where();
    pow.restore_where();
    prod.restore_where();
    multiply_fiber_inplace.restore_where();
    multiply_fiber.restore_where();
    multiply_inplace.restore_where();
    multiply_slice.restore_where();
    randn.restore_where();
    relu.restore_where();
    relu_backward.restore_where();
    relu_forward.restore_where();
    rope.restore_where();
    rope_backward.restore_where();
    scale.restore_where();
    scale_inplace.restore_where();
    silu_backward.restore_where();
    silu_forward.restore_where();
    softmax.restore_where();
    softmax_inplace.restore_where();
    sqrt.restore_where();
    sqrt_inplace.restore_where();
    subcopy.restore_where();
    subtract_indexed_outputs.restore_where();
    sum_fiber.restore_where();
    sum_slice.restore_where();
    sumprod_fiber.restore_where();
    sumprod_slice.restore_where();
    total_sum_accum.restore_where();
    transpose.restore_where();
}

} // namespace nntile
