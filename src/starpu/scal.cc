/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/scal.cc
 * Scal operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/scal.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/scal.hh"
#include "nntile/starpu/clear.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Scal<std::tuple<T>>::Scal():
    codelet("nntile_scal", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::scal::cpu<T>
template<typename T>
void Scal<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
    kernel::scal::cpu<T>(args->nelems, args->alpha, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Scal<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scal<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Scal<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::scal::cuda<T>
template<typename T>
void Scal<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::scal::cuda<T>(stream, args->nelems, args->alpha, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Scal<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scal<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Scal<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Scal<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for scal tasks that depends only on cl_arg
template<typename T>
uint32_t Scal<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Scal<std::tuple<T>>::submit(
    Index nelems, Scalar alpha, Handle src, Handle dst)
{
    constexpr Scalar zero = 0.0;
    // if alpha is zero,sfunction reduces to clear
    if(alpha == zero)
    {
        clear.submit(dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_W, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in scal task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Scal<std::tuple<nntile::fp64_t>>;
template class Scal<std::tuple<nntile::fp32_t>>;
template class Scal<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Scal<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Scal<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Scal<std::tuple<nntile::bf16_t>>;

//! Pack of scal operations for different types
scal_pack_t scal;

} // namespace nntile::starpu
