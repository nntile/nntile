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

#include "nntile/starpu/log_scalar.hh"
#ifndef STARPU_SIMGRID
#include "nntile/logger/logger_thread.hh"
#endif // STARPU_SIMGRID
#include <cstdlib>

//! StarPU wrappers to log scalar
namespace nntile::starpu::log_scalar
{

//! Apply log_scalar operation for StarPU buffers in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
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

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_log_scalar_fp32",
            nullptr,
            {cpu<fp32_t>},
            {}
            );

    codelet_bf16.init("nntile_log_scalar_bf16",
            nullptr,
            {cpu<bf16_t>},
            {}
            );

    codelet_fp32_fast_tf32.init("nntile_log_scalar_fp32_fast_tf32",
            nullptr,
            {cpu<fp32_t>},
            {}
            );

    codelet_fp64.init("nntile_log_scalar_fp64",
            nullptr,
            {cpu<fp64_t>},
            {}
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(const std::string &name, Handle value)
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
    args_t *args = new args_t;
    args->name = name;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
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
template
void submit<fp32_t>(const std::string &name, Handle value);

template
void submit<fp64_t>(const std::string &name, Handle value);

template
void submit<fp32_fast_tf32_t>(const std::string &name, Handle value);

template
void submit<bf16_t>(const std::string &name, Handle value);

} // namespace nntile::starpu::log_scalar
