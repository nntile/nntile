/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot_scalar_inverse.cc
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/hypot_scalar_inverse.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_scalar_inverse.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
HypotScalarInverse<std::tuple<T>>::HypotScalarInverse():
    codelet("nntile_hypot_scalar_inverse", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot_scalar_inverse operation for StarPU buffers in CPU
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *dst = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::hypot_scalar_inverse::cpu<T>(args->nelems, args->eps, args->alpha,
            dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotScalarInverse<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot_scalar_inverse for StarPU buffers on CUDA
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *dst = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::hypot_scalar_inverse::cuda<T>(stream, args->nelems, args->eps,
            args->alpha, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void HypotScalarInverse<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void HypotScalarInverse<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    HypotScalarInverse<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t HypotScalarInverse<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot_scalar_inverse task
template<typename T>
void HypotScalarInverse<std::tuple<T>>::submit(
        Index nelems, Scalar eps, Scalar alpha, Handle dst)
//! Insert hypot_scalar_inverse task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->eps = eps;
    args->alpha = alpha;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in hypot_scalar_inverse task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class HypotScalarInverse<std::tuple<nntile::fp64_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_tf32_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_fp16_t>>;
template class HypotScalarInverse<std::tuple<nntile::fp32_fast_bf16_t>>;
template class HypotScalarInverse<std::tuple<nntile::bf16_t>>;

//! Pack of hypot_scalar_inverse operations for different types
hypot_scalar_inverse_pack_t hypot_scalar_inverse;

} // namespace nntile::starpu
