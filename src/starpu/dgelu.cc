/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/dgelu.cc
 * Derivative of GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/dgelu.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// NNTile headers
#include "nntile/kernel/dgelu.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
DGelu<std::tuple<T>>::DGelu():
    codelet("nntile_dgelu", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply dgelu on StarPU buffer on CPU
template<typename T>
void DGelu<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::dgelu::cpu<T>(args->nelems, data);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void DGelu<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void DGelu<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void DGelu<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply dgelu on StarPU buffer on CUDA
template<typename T>
void DGelu<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::dgelu::cuda<T>(stream, args->nelems, data);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void DGelu<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void DGelu<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void DGelu<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    DGelu<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t DGelu<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void DGelu<std::tuple<T>>::submit(Index nelems, Handle data)
{
    // Get arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Submit task
    //double nflops = 5 * nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, data.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in dgelu task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class DGelu<std::tuple<nntile::fp64_t>>;
template class DGelu<std::tuple<nntile::fp32_t>>;
template class DGelu<std::tuple<nntile::fp32_fast_tf32_t>>;
template class DGelu<std::tuple<nntile::fp32_fast_fp16_t>>;
template class DGelu<std::tuple<nntile::fp32_fast_bf16_t>>;
template class DGelu<std::tuple<nntile::bf16_t>>;

//! Pack of dgelu operations for different types
dgelu_pack_t dgelu;

} // namespace nntile::starpu
