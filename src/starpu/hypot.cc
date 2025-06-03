/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot.cc
 * hypot operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/hypot.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Hypot<std::tuple<T>>::Hypot():
    codelet("nntile_hypot", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply hypot operation for StarPU buffers in CPU
template<typename T>
void Hypot<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
    kernel::hypot::cpu<T>(args->nelems, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot for StarPU buffers on CUDA
template<typename T>
void Hypot<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::hypot::cuda<T>(stream, args->nelems, args->alpha, src,
            args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Hypot<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Hypot<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Hypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t Hypot<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit hypot task
template<typename T>
void Hypot<std::tuple<T>>::submit(
        Index nelems, Scalar alpha, Handle src, Scalar beta, Handle dst)
//! Insert hypot task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0, one = 1;
    // If beta is zero this function reduces to scal
    //if(beta == zero)
    //{
    //    throw std::runtime_error("modulus(apha*src) is not yet implemented");
    //    //scal::submit<T>(nelems, alpha, src, dst);
    //    return;
    //}
    // If beta is non-zero and alpha is zero then reduce to scal_inplace
    //if(alpha == zero)
    //{
    //    throw std::runtime_error("modulus_inplace(beta*dst) is not yet "
    //            "implemented");
    //    //scal_inplace::submit<T>(nelems, beta, dst);
    //    return;
    //}
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == one)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, dst.get(), 0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in hypot task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Hypot<std::tuple<nntile::fp64_t>>;
template class Hypot<std::tuple<nntile::fp32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Hypot<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Hypot<std::tuple<nntile::bf16_t>>;

//! Pack of hypot operations for different types
hypot_pack_t hypot;

} // namespace nntile::starpu
