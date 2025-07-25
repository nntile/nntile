/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/maxsumexp.cc
 * Max and sum of exponents for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/maxsumexp.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/maxsumexp.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
MaxSumExp<std::tuple<T>>::MaxSumExp():
    codelet("nntile_maxsumexp", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Max and sum of exponents along middle axis of StarPU buffer on CPU
template<typename T>
void MaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
    kernel::maxsumexp::cpu<T>(args->m, args->n, args->k, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void MaxSumExp<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Max and sum of exponents along middle axis of StarPU buffer on CUDA
template<typename T>
void MaxSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::maxsumexp::cuda<T>(stream, args->m, args->n, args->k, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void MaxSumExp<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void MaxSumExp<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    MaxSumExp<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for maxsumexp tasks that depends only on m, n and k
template<typename T>
uint32_t MaxSumExp<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void MaxSumExp<std::tuple<T>>::submit(Index m, Index n, Index k, Handle src, Handle dst, int redux)
//! Insert maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(redux != 0)
    {
        dst_mode = STARPU_REDUX;
    }
    else
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * m * (k+2) * n;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, dst.get(),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in maxsumexp task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class MaxSumExp<std::tuple<nntile::fp64_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_tf32_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_fp16_t>>;
template class MaxSumExp<std::tuple<nntile::fp32_fast_bf16_t>>;
template class MaxSumExp<std::tuple<nntile::bf16_t>>;

//! Pack of maxsumexp operations for different types
maxsumexp_pack_t maxsumexp;

} // namespace nntile::starpu
