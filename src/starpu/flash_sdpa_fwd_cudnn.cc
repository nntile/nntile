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

// Other NNTile headers
#include "nntile/kernel/flash_sdpa_fwd_cudnn.hh"

//! StarPU wrappers for flash_sdpa_fwd_cudnn operation
namespace nntile::starpu::flash_sdpa_fwd_cudnn
{

//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CPU
template<typename T>
void KernelWrapper<T>::cpu(void *buffers[], void *cl_args)
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
    kernel::flash_sdpa_fwd_cudnn::cpu<T>(args->m, args->n, args->k, args->alpha,
            src, args->beta, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply flash_sdpa_fwd_cudnn on StarPU buffer on CUDA
template<typename T>
void KernelWrapper<T>::cuda(void *buffers[], void *cl_args)
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
    kernel::flash_sdpa_fwd_cudnn::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Define codelet pack
codelet_pack_t codelet_pack = codelet_pack_t(
    "nntile_flash_sdpa_fwd_cudnn",
    nullptr
).set_modes_fixed({STARPU_R, STARPU_RW});

template<typename T>
void submit(Index m, Index n, Index k, Scalar alpha, Handle src, Scalar beta,
        Handle dst)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(codelet_pack.get_codelet_ptr<T>(),
            STARPU_R, src.get(),
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in flash_sdpa_fwd_cudnn task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index m, Index n, Index k, Scalar alpha, Handle src,
        Scalar beta, Handle dst);

template
void submit<fp64_t>(Index m, Index n, Index k, Scalar alpha, Handle src,
        Scalar beta, Handle dst);

} // namespace nntile::starpu::flash_sdpa_fwd_cudnn
