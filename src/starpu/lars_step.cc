/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/lars_step.cc
 * Fused LARS step operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/lars_step.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/lars_step.hh"

//! StarPU wrappers for one step of LARS optimizer
namespace nntile::starpu
{

//! Constructor
template<typename T>
LarsStep<std::tuple<T>>::LarsStep():
    codelet("nntile_lars_step", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply LARS step on StarPU buffers on CPU
template<typename T>
void LarsStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T* p = interfaces[1]->get_ptr<T>();
    T* weight_norm = interfaces[2]->get_ptr<T>();
    T* grad_norm = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::lars_step::cpu<T>(
        args->num_elems,
        args->lr,
        args->trust_ratio,
        weight_norm[0],
        grad_norm[0],
        args->weight_decay,
        grad,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void LarsStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LarsStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void LarsStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply LARS step operation on StarPU buffer on CUDA
template<typename T>
void LarsStep<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T* p = interfaces[1]->get_ptr<T>();
    T* weight_norm = interfaces[2]->get_ptr<T>();
    T* grad_norm = interfaces[3]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::lars_step::cuda<T>(
        stream,
        args->num_elems,
        args->lr,
        args->trust_ratio,
        Scalar(*weight_norm),
        Scalar(*grad_norm),
        args->weight_decay,
        grad,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void LarsStep<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LarsStep<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void LarsStep<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    LarsStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for lars_step tasks that depends only on cl_arg
template<typename T>
uint32_t LarsStep<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->num_elems, sizeof(args->num_elems), hash);
    return hash;
}

//! Submit LARS step task
template<typename T>
void LarsStep<std::tuple<T>>::submit(
    Index num_elems,
    Scalar lr,
    Scalar trust_ratio,
    Scalar weight_decay,
    Handle grad,
    Handle param,
    Handle weight_norm,
    Handle grad_norm
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_elems = num_elems;
    args->lr = lr;
    args->trust_ratio = trust_ratio;
    args->weight_decay = weight_decay;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, grad.get(),
            STARPU_RW, param.get(),
            STARPU_R, weight_norm.get(),
            STARPU_R, grad_norm.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in lars_step task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class LarsStep<std::tuple<nntile::fp64_t>>;
template class LarsStep<std::tuple<nntile::fp32_t>>;
template class LarsStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class LarsStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class LarsStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class LarsStep<std::tuple<nntile::bf16_t>>;
template class LarsStep<std::tuple<nntile::fp16_t>>;

//! Pack of lars_step operations for different types
lars_step_pack_t lars_step;

} // namespace nntile::starpu
