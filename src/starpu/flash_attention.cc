/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/flash_attention.cc
 * Flash attention operation for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/flash_attention.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/flash_attention.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
FlashAttention<std::tuple<T>>::FlashAttention():
    codelet("nntile_flash_attention", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::flash_attention::cpu<T>
template<typename T>
void FlashAttention<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *Q = interfaces[0]->get_ptr<T>();
    const T *K = interfaces[1]->get_ptr<T>();
    const T *V = interfaces[2]->get_ptr<T>();
    T *O = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::flash_attention::cpu<T>(args->batch, args->num_heads,
            args->seq_len, args->head_dim, Q, K, V, args->scale, O);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void FlashAttention<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void FlashAttention<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void FlashAttention<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::flash_attention::cuda<T>
template<typename T>
void FlashAttention<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *Q = interfaces[0]->get_ptr<T>();
    const T *K = interfaces[1]->get_ptr<T>();
    const T *V = interfaces[2]->get_ptr<T>();
    T *O = interfaces[3]->get_ptr<T>();
    T *logsumexp = interfaces[4]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::flash_attention::cuda<T>(stream, args->batch, args->num_heads,
            args->seq_len, args->head_dim, Q, K, V, args->scale, O, logsumexp);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void FlashAttention<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void FlashAttention<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void FlashAttention<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[],
        void *cl_args)
    noexcept
{
    // Fall back to FP32
    FlashAttention<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for flash_attention tasks
template<typename T>
uint32_t FlashAttention<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    hash = starpu_hash_crc32c_be_n(&args->num_heads, sizeof(args->num_heads), hash);
    hash = starpu_hash_crc32c_be_n(&args->seq_len, sizeof(args->seq_len), hash);
    hash = starpu_hash_crc32c_be_n(&args->head_dim, sizeof(args->head_dim), hash);
    return hash;
}

template<typename T>
void FlashAttention<std::tuple<T>>::submit(Index batch, Index num_heads,
        Index seq_len, Index head_dim, Scalar scale, Handle Q, Handle K,
        Handle V, Handle O, Handle logsumexp)
//! Insert flash_attention task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routine
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->batch = batch;
    args->num_heads = num_heads;
    args->seq_len = seq_len;
    args->head_dim = head_dim;
    args->scale = scale;
    // Estimate FLOPs: 2 * batch * num_heads * seq_len^2 * head_dim (QK^T + softmax@V)
    double nflops = 2.0 * batch * num_heads * seq_len * seq_len * head_dim;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, Q.get(),
            STARPU_R, K.get(),
            STARPU_R, V.get(),
            STARPU_W, O.get(),
            STARPU_W, logsumexp.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_attention task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class FlashAttention<std::tuple<nntile::fp64_t>>;
template class FlashAttention<std::tuple<nntile::fp32_t>>;
template class FlashAttention<std::tuple<nntile::fp32_fast_tf32_t>>;
template class FlashAttention<std::tuple<nntile::fp32_fast_fp16_t>>;
template class FlashAttention<std::tuple<nntile::fp32_fast_bf16_t>>;
template class FlashAttention<std::tuple<nntile::bf16_t>>;
template class FlashAttention<std::tuple<nntile::fp16_t>>;

//! Pack of flash_attention operations for different types
flash_attention_pack_t flash_attention;

} // namespace nntile::starpu
