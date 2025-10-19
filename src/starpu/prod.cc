/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/prod.cc
 * Per-element product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/multiply.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/multiply.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Prod<std::tuple<T>>::Prod():
    codelet("nntile_prod", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::prod::cpu<T>
template<typename T>
void Prod<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::prod::cpu<T>(nelems, src1, src2, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Prod<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Prod<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Prod<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply prod on StarPU buffer on CUDA
template<typename T>
void Prod<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::prod::cuda<T>(stream, nelems, src1, src2, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Prod<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Prod<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Prod<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Prod<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t Prod<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Prod<std::tuple<T>>::submit(Index nelems, Handle src1, Handle src2, Handle dst)
{
    Index *nelems_ = new Index{nelems};
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src1.get(),
            STARPU_R, src2.get(),
            STARPU_W, dst.get(),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in prod task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Prod<std::tuple<nntile::fp64_t>>;
template class Prod<std::tuple<nntile::fp32_t>>;
template class Prod<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Prod<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Prod<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Prod<std::tuple<nntile::bf16_t>>;

//! Pack of prod operations for different types
prod_pack_t prod;

} // namespace nntile::starpu
