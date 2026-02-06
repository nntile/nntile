/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelutanh.cc
 * Approximate GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding headers
#include "nntile/starpu/gelutanh.hh"

// Standard libraries
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/gelutanh.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
GeluTanh<std::tuple<T>>::GeluTanh():
    codelet("nntile_gelutanh", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply gelutanh on StarPU buffer on CPU
template<typename T>
void GeluTanh<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
    kernel::gelutanh::cpu<T>(args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluTanh<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanh<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void GeluTanh<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply gelutanh on StarPU buffer on CUDA
template<typename T>
void GeluTanh<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::gelutanh::cuda<T>(stream, args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void GeluTanh<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanh<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void GeluTanh<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    GeluTanh<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t GeluTanh<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit gelutanh task
template<typename T>
void GeluTanh<std::tuple<T>>::submit(
    Index nelems,
    Handle src,
    Handle dst
)
//! Insert gelutanh task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    //double nflops = 5 * nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_W, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelutanh task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class GeluTanh<std::tuple<nntile::fp64_t>>;
template class GeluTanh<std::tuple<nntile::fp32_t>>;
template class GeluTanh<std::tuple<nntile::fp32_fast_tf32_t>>;
template class GeluTanh<std::tuple<nntile::fp32_fast_fp16_t>>;
template class GeluTanh<std::tuple<nntile::fp32_fast_bf16_t>>;
template class GeluTanh<std::tuple<nntile::bf16_t>>;
template class GeluTanh<std::tuple<nntile::fp16_t>>;

//! Pack of gelutanh operations for different types
gelutanh_pack_t gelutanh;

} // namespace nntile::starpu
