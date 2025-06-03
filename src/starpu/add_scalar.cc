/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_scalar.cc
 * Add_scalar operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/add_scalar.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_scalar.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AddScalar<std::tuple<T>>::AddScalar():
    codelet("nntile_add_scalar", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply add_scalar for StarPU buffer in CPU
template<typename T>
void AddScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *dst = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::add_scalar::cpu<T>(
        args->nelems,
        args->alpha,
        args->beta,
        dst
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddScalar<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddScalar<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddScalar<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add_scalar for StarPU buffers on CUDA
template<typename T>
void AddScalar<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::add_scalar::cuda<T>(
        stream,
        args->nelems,
        args->alpha,
        args->beta,
        dst
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddScalar<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddScalar<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddScalar<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddScalar<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_scalar tasks
template<typename T>
uint32_t AddScalar<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit add_scalar task
template<typename T>
void AddScalar<std::tuple<T>>::submit(
    Index nelems,
    Scalar alpha,
    Scalar beta,
    Handle dst
)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = 2 * nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, dst.get(),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add_scalar task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddScalar<std::tuple<nntile::fp64_t>>;
template class AddScalar<std::tuple<nntile::fp32_t>>;
template class AddScalar<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddScalar<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddScalar<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddScalar<std::tuple<nntile::bf16_t>>;

//! Pack of add_scalar operations for different types
add_scalar_pack_t add_scalar;

} // namespace nntile::starpu
