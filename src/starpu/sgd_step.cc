/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sgd_step.cc
 * Fused SGD with momentum step operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/sgd_step.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sgd_step.hh"

//! StarPU wrappers for one step of SGD with momentum optimizer
namespace nntile::starpu
{

//! Constructor
template<typename T>
SGDStep<std::tuple<T>>::SGDStep():
    codelet("nntile_sgd_step", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply SGD step on StarPU buffers on CPU
template<typename T>
void SGDStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T *velocity = interfaces[1]->get_ptr<T>();
    T* p = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::sgd_step::cpu<T>(
        args->num_iter,
        args->num_elems,
        args->momentum,
        args->lr,
        args->weight_decay,
        args->dampening,
        args->nesterov,
        grad,
        velocity,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void SGDStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply SGD step operation on StarPU buffer on CUDA
template<typename T>
void SGDStep<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T *velocity = interfaces[1]->get_ptr<T>();
    T* p = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::sgd_step::cuda<T>(
        stream,
        args->num_iter,
        args->num_elems,
        args->momentum,
        args->lr,
        args->weight_decay,
        args->dampening,
        args->nesterov,
        grad,
        velocity,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SGDStep<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SGDStep<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SGDStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sgd_step tasks that depends only on cl_arg
template<typename T>
uint32_t SGDStep<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    bool first_iter = args->num_iter == 1;
    hash = starpu_hash_crc32c_be_n(&first_iter, sizeof(bool), hash);
    hash = starpu_hash_crc32c_be_n(&args->num_elems, sizeof(args->num_elems), hash);
    hash = starpu_hash_crc32c_be_n(&args->momentum, sizeof(args->momentum), hash);
    hash = starpu_hash_crc32c_be_n(&args->weight_decay, sizeof(args->weight_decay), hash);
    hash = starpu_hash_crc32c_be_n(&args->dampening, sizeof(args->dampening), hash);
    hash = starpu_hash_crc32c_be_n(&args->nesterov, sizeof(args->nesterov), hash);
    return hash;
}

//! Submit SGD step task
template<typename T>
void SGDStep<std::tuple<T>>::submit(
    Index num_iter,
    Index num_elems,
    Scalar momentum,
    Scalar lr,
    Scalar weight_decay,
    Scalar dampening,
    bool nesterov,
    Handle grad,
    Handle velocity,
    Handle param
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_iter = num_iter;
    args->num_elems = num_elems;
    args->momentum = momentum;
    args->lr = lr;
    args->weight_decay = weight_decay;
    args->dampening = dampening;
    args->nesterov = nesterov;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, grad.get(),
            STARPU_RW, velocity.get(),
            STARPU_RW, param.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sgd_step task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SGDStep<std::tuple<nntile::fp64_t>>;
template class SGDStep<std::tuple<nntile::fp32_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SGDStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SGDStep<std::tuple<nntile::bf16_t>>;
template class SGDStep<std::tuple<nntile::fp16_t>>;

//! Pack of sgd_step operations for different types
sgd_step_pack_t sgd_step;

} // namespace nntile::starpu
