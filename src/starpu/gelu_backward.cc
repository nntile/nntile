/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu_backward.cc
 * GeLU backward operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/gelu_backward.hh"

// Standard libraries
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/gelu_backward.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
GeluBackward<std::tuple<T>>::GeluBackward():
    codelet("nntile_gelu_backward", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply gelu_backward on StarPU buffer on CPU
template<typename T>
void GeluBackward<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *x = interfaces[0]->get_ptr<T>();
    const T *dy = interfaces[1]->get_ptr<T>();
    T *dx = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::gelu_backward::cpu<T>(args->nelems, x, dy, dx);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluBackward<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply gelu_backward on StarPU buffer on CUDA
template<typename T>
void GeluBackward<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *x = interfaces[0]->get_ptr<T>();
    const T *dy = interfaces[1]->get_ptr<T>();
    T *dx = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::gelu_backward::cuda<T>(stream, args->nelems, x, dy, dx);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluBackward<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluBackward<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluBackward<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t GeluBackward<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void GeluBackward<std::tuple<T>>::submit(Index nelems, Handle x, Handle dy, Handle dx)
{
    // Codelet arguments
    Index *nelems_ = (Index *)std::malloc(sizeof(*nelems_));
    *nelems_ = nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, x.get(),
            STARPU_R, dy.get(),
            STARPU_RW, dx.get(),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelu_backward task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluBackward<std::tuple<nntile::fp64_t>>;
template class GeluBackward<std::tuple<nntile::fp32_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluBackward<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluBackward<std::tuple<nntile::bf16_t>>;

//! Pack of gelu_backward operations for different types
gelu_backward_pack_t gelu_backward;

} // namespace nntile::starpu
