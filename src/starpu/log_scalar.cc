/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/log_scalar.cc
 * StarPU wrapper to log scalar value
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/log_scalar.hh"

// Standard headers
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/logger/logger_thread.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
LogScalar<std::tuple<T>>::LogScalar():
    codelet("nntile_log_scalar", nullptr, cpu_funcs, cuda_funcs)
{
}

//! Apply log_scalar operation for StarPU buffers in CPU
template<typename T>
void LogScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *value = interfaces[0]->get_ptr<T>();
    // Launch logger
    using Y = typename T::repr_t;
    logger::log_scalar(args->name, static_cast<Y>(*value));
    // Call destructor explicitly, since the object is C++ object
    delete args;
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void LogScalar<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogScalar<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LogScalar<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LogScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

//! Submit log_scalar task
template<typename T>
void LogScalar<std::tuple<T>>::submit(const std::string &name, Handle value)
//! Insert log_scalar task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Get out if logger thread is not running
    if(not nntile::logger::logger_running)
    {
        return;
    }
    // Codelet arguments
    args_t *args = new args_t();
    args->name = name;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, value.get(),
            STARPU_CL_ARGS_NFREE, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in log_scalar task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class LogScalar<std::tuple<nntile::fp64_t>>;
template class LogScalar<std::tuple<nntile::fp32_t>>;
template class LogScalar<std::tuple<nntile::fp32_fast_tf32_t>>;
template class LogScalar<std::tuple<nntile::fp32_fast_fp16_t>>;
template class LogScalar<std::tuple<nntile::fp32_fast_bf16_t>>;
template class LogScalar<std::tuple<nntile::bf16_t>>;

//! Pack of log_scalar operations for different types
log_scalar_pack_t log_scalar;

} // namespace nntile::starpu
