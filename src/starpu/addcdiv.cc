/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/addcdiv.cc
 * Per-element addcdiv operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/addcdiv.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/addcdiv.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AddCdiv<std::tuple<T>>::AddCdiv():
    codelet("nntile_addcdiv", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply addcdiv operation on StarPU buffers on CPU
template<typename T>
void AddCdiv<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *nom = interfaces[0]->get_ptr<T>();
    const T *denom = interfaces[1]->get_ptr<T>();
    T *src = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::addcdiv::cpu<T>(
        args->val, args->eps, args->nelems, nom, denom, src);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddCdiv<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddCdiv<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddCdiv<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply addcdiv on StarPU buffer on CUDA
template<typename T>
void AddCdiv<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *nom = interfaces[0]->get_ptr<T>();
    const T *denom = interfaces[1]->get_ptr<T>();
    T *src = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::addcdiv::cuda<T>(
        stream, args->val, args->eps, args->nelems, nom, denom, src);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddCdiv<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddCdiv<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddCdiv<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddCdiv<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t AddCdiv<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void AddCdiv<std::tuple<T>>::submit(
    Index nelems,
    Scalar val,
    Scalar eps,
    Handle nom,
    Handle denom,
    Handle src
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->val = val;
    args->eps = eps;
    args->nelems = nelems;
    //double nflops = 5 * nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, nom.get(),
            STARPU_R, denom.get(),
            STARPU_RW, src.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in addcdiv task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddCdiv<std::tuple<nntile::fp64_t>>;
template class AddCdiv<std::tuple<nntile::fp32_t>>;
template class AddCdiv<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddCdiv<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddCdiv<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddCdiv<std::tuple<nntile::bf16_t>>;

//! Pack of addcdiv operations for different types
addcdiv_pack_t addcdiv;

} // namespace nntile::starpu
