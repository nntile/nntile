/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/maximum.cc
 * StarPU wrappers for maximum operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/maximum.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/maximum.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Maximum<std::tuple<T>>::Maximum():
    codelet("nntile_maximum", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply maximum operation on StarPU buffers on CPU
template<typename T>
void Maximum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::maximum::cpu<T>(args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Maximum<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Maximum<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Maximum<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply per element maximum on StarPU buffer on CUDA
template<typename T>
void Maximum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::maximum::cuda<T>(stream, args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Maximum<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Maximum<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Maximum<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Maximum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t Maximum<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Maximum<std::tuple<T>>::submit(Index nelems, Handle src, Handle dst)
{
    args_t *args = (args_t *)malloc(sizeof(*args));
    args->nelems = nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in maximum task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Maximum<std::tuple<nntile::fp64_t>>;
template class Maximum<std::tuple<nntile::fp32_t>>;
template class Maximum<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Maximum<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Maximum<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Maximum<std::tuple<nntile::bf16_t>>;

//! Pack of maximum operations for different types
maximum_pack_t maximum;

} // namespace nntile::starpu
