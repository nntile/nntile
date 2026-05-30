/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sum.cc
 * Sum all elements of a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/sum.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sum.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Sum<std::tuple<T>>::Sum():
    codelet("nntile_sum", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sum::cpu<T>
template<typename T>
void Sum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::sum::cpu<T>(args->nelems, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Sum<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sum::cuda<T>
template<typename T>
void Sum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::sum::cuda<T>(stream, args->nelems, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Sum<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Sum<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Sum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sum tasks
template<typename T>
uint32_t Sum<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters nelems
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    hash = starpu_hash_crc32c_be_n(&args->alpha, sizeof(args->alpha), hash);
    hash = starpu_hash_crc32c_be_n(&args->beta, sizeof(args->beta), hash);
    return hash;
}

template<typename T>
void Sum<std::tuple<T>>::submit(Index nelems, Scalar alpha, Handle src,
        Scalar beta, Handle dst, int redux)
//! Insert sum task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Set destination access mode
    starpu_data_access_mode dst_mode;
    if(beta == 0.0)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * (nelems+1);
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            dst_mode, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sum task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Sum<std::tuple<nntile::fp64_t>>;
template class Sum<std::tuple<nntile::fp32_t>>;
template class Sum<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Sum<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Sum<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Sum<std::tuple<nntile::bf16_t>>;
template class Sum<std::tuple<nntile::fp16_t>>;

//! Pack of sum operations for different types
sum_pack_t sum;

} // namespace nntile::starpu
