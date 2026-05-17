/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/isfinite.cc
 * Accumulate flags for NaN and Inf in a buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/isfinite.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/isfinite.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Isfinite<std::tuple<T>>::Isfinite():
    codelet("nntile_isfinite", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::isfinite::cpu<T>
template<typename T>
void Isfinite<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    bool_t *flag = interfaces[1]->get_ptr<bool_t>();
    // Launch kernel
    kernel::isfinite::cpu<T>(args->nelems, data, flag);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Isfinite<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::isfinite::cuda<T>
template<typename T>
void Isfinite<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    bool_t *flag = interfaces[1]->get_ptr<bool_t>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::isfinite::cuda<T>(stream, args->nelems, data, flag);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Isfinite<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<bf16_t>>::cuda(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Isfinite<std::tuple<fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Isfinite<std::tuple<fp16_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t Isfinite<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Isfinite<std::tuple<T>>::submit(Index nelems, Handle data, Handle flag)
//! Insert isfinite task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, data.get(),
            STARPU_RW, flag.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in isfinite task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Isfinite<std::tuple<nntile::fp64_t>>;
template class Isfinite<std::tuple<nntile::fp32_t>>;
template class Isfinite<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Isfinite<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Isfinite<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Isfinite<std::tuple<nntile::bf16_t>>;
template class Isfinite<std::tuple<nntile::fp16_t>>;

//! Pack of pow operations for different types
isfinite_pack_t isfinite;

} // namespace nntile::starpu
