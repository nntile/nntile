/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/adam_step.cc
 * Per-element addcdiv operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/adam_step.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/adam_step.hh"

//! StarPU wrappers for one step of Adam optimizer
namespace nntile::starpu
{

//! Constructor
template<typename T>
AdamStep<std::tuple<T>>::AdamStep():
    codelet("nntile_adam_step", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply Adam step on StarPU buffers on CPU
template<typename T>
void AdamStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T *first_moments = interfaces[1]->get_ptr<T>();
    T *second_moments = interfaces[2]->get_ptr<T>();
    T* p = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::adam_step::cpu<T>(
        args->num_iter,
        args->num_elems,
        args->beta_1,
        args->beta_2,
        args->eps,
        args->lr,
        args->weight_decay,
        grad,
        first_moments,
        second_moments,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AdamStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply Adam step operation on StarPU buffer on CUDA
template<typename T>
void AdamStep<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *grad = interfaces[0]->get_ptr<T>();
    T *first_moments = interfaces[1]->get_ptr<T>();
    T *second_moments = interfaces[2]->get_ptr<T>();
    T* p = interfaces[3]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::adam_step::cuda<T>(
        stream,
        args->num_iter,
        args->num_elems,
        args->beta_1,
        args->beta_2,
        args->eps,
        args->lr,
        args->weight_decay,
        grad,
        first_moments,
        second_moments,
        p
    );
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void AdamStep<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for adam_step tasks that depends only on cl_arg
template<typename T>
uint32_t AdamStep<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->num_elems, sizeof(args->num_elems), hash);
    return hash;
}

//! Submit Adam step task
template<typename T>
void AdamStep<std::tuple<T>>::submit(
    Index num_iter,
    Index num_elems,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    Handle grad,
    Handle first_moment,
    Handle second_moment,
    Handle param
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_iter = num_iter;
    args->num_elems = num_elems;
    args->beta_1 = beta_1;
    args->beta_2 = beta_2;
    args->eps = eps;
    args->lr = lr;
    args->weight_decay = weight_decay;
    //double nflops = 5 * nelems;
    // Submit task
    enum starpu_data_access_mode moments_mode;
    if (num_iter == 1)
    {
        moments_mode = STARPU_W;
    }
    else
    {
        moments_mode = STARPU_RW;
    }
    int ret = starpu_task_insert(&codelet,
            STARPU_R, grad.get(),
            moments_mode, first_moment.get(),
            moments_mode, second_moment.get(),
            STARPU_RW, param.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in adam_step task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AdamStep<std::tuple<nntile::fp64_t>>;
template class AdamStep<std::tuple<nntile::fp32_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AdamStep<std::tuple<nntile::bf16_t>>;

//! Pack of adam_step operations for different types
adam_step_pack_t adam_step;

} // namespace nntile::starpu
