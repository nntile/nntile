/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/flash_sdpa_fwd_cudnn.cc
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Third-party libraries
#ifdef NNTILE_USE_CUDA
#include <cudnn.h>
#endif

// Other NNTile headers
#include "nntile/kernel/flash_sdpa_fwd_cudnn.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
FlashSdpaFwdCudnn<std::tuple<T>>::FlashSdpaFwdCudnn():
    codelet("nntile_flash_sdpa_fwd_cudnn", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CPU (not supported)
template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // CPU implementation not supported for cuDNN Flash Attention
    // This operation requires CUDA/cuDNN
    // Do nothing - this codelet should only be called on CUDA devices
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CUDA
template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    args_t *args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *K = interfaces[0]->get_ptr<T>();           // Key
    const T *Q = interfaces[1]->get_ptr<T>();           // Query
    const T *mask = interfaces[2]->get_ptr<T>();        // Mask
    T *logsumexp = interfaces[3]->get_ptr<T>();         // Log-sum-exp
    const T *V = interfaces[4]->get_ptr<T>();           // Value
    T *A = interfaces[5]->get_ptr<T>();                 // Attention output
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Create cuDNN handle and set stream
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle); // TODO: handle is from starpu_... call
    if (status != CUDNN_STATUS_SUCCESS) {
        return; // Fail silently in noexcept function
    }
    status = cudnnSetStream(handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        // Attempt cleanup even if we're about to fail
        (void)cudnnDestroy(handle);
        return; // Fail silently in noexcept function
    }
    // Launch kernel
    kernel::flash_sdpa_fwd_cudnn::cuda<T>(handle, args->seq, args->head,
            args->batch, K, Q, mask, logsumexp, V, A);
    // Cleanup cuDNN handle
    (void)cudnnDestroy(handle); // Explicitly ignore return value in noexcept context
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for flash_sdpa_fwd_cudnn tasks that depends only on cl_arg
template<typename T>
uint32_t FlashSdpaFwdCudnn<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->seq, sizeof(args->seq), hash);
    hash = starpu_hash_crc32c_be_n(&args->head, sizeof(args->head), hash);
    hash = starpu_hash_crc32c_be_n(&args->batch, sizeof(args->batch), hash);
    return hash;
}

template<typename T>
void FlashSdpaFwdCudnn<std::tuple<T>>::submit(Index seq, Index head, Index batch,
        Handle K, Handle Q, Handle mask, Handle logsumexp, Handle V, Handle A)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->seq = seq;
    args->head = head;
    args->batch = batch;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, K.get(),              // Key
            STARPU_R, Q.get(),              // Query
            STARPU_R, mask.get(),           // Mask
            STARPU_RW, logsumexp.get(),     // Log-sum-exp
            STARPU_R, V.get(),              // Value
            STARPU_RW, A.get(),             // Attention output
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_sdpa_fwd_cudnn task submission");
    }
}

// Explicit instantiation
template class FlashSdpaFwdCudnn<std::tuple<nntile::bf16_t>>;
template class FlashSdpaFwdCudnn<std::tuple<nntile::fp16_t>>;

//! Pack of flash_sdpa_fwd_cudnn operations for different types
flash_sdpa_fwd_cudnn_pack_t flash_sdpa_fwd_cudnn;

} // namespace nntile::starpu
