/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/transpose.cc
 * Transpose operation for StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/transpose.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/transpose.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Transpose<std::tuple<T>>::Transpose():
    codelet("nntile_transpose", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::transpose::cpu<T>
template<typename T>
void Transpose<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::transpose::cpu<T>(args->m, args->n, args->alpha, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::transpose::cuda<T>
template<typename T>
void Transpose<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
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
    kernel::transpose::cuda<T>(stream, args->m, args->n, args->alpha, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Transpose<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Transpose<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Transpose<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for transpose tasks
template<typename T>
uint32_t Transpose<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    return hash;
}

template<typename T>
void Transpose<std::tuple<T>>::submit(Index m, Index n, Scalar alpha, Handle src, Handle dst)
//! Insert transpose task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->alpha = alpha;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 2 * m * n;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_W, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in transpose task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Transpose<std::tuple<nntile::fp64_t>>;
template class Transpose<std::tuple<nntile::fp32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Transpose<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Transpose<std::tuple<nntile::bf16_t>>;

//! Pack of transpose operations for different types
transpose_pack_t transpose;

} // namespace nntile::starpu
