/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/add.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add.hh"
#include "nntile/starpu/scal.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Add<std::tuple<T>>::Add():
    codelet("nntile_add", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply add operation for StarPU buffers in CPU
template<typename T>
void Add<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::add::cpu<T>(
        args->nelems, args->alpha, src1, args->beta, src2, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply add for StarPU buffers on CUDA
template<typename T>
void Add<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src1 = interfaces[0]->get_ptr<T>();
    const T *src2 = interfaces[1]->get_ptr<T>();
    T *dst = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::add::cuda<T>(stream, args->nelems, args->alpha, src1, args->beta,
            src2, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void Add<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void Add<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Add<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add tasks that depends only on cl_arg
template<typename T>
uint32_t Add<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

template<typename T>
void Add<std::tuple<T>>::submit(
    Index nelems,
    Scalar alpha,
    Handle src1,
    Scalar beta,
    Handle src2,
    Handle dst
)
{
    constexpr Scalar zero = 0;
    // If beta is zero this function reduces to scal_inplace
    if(beta == zero)
    {
        // dst = alpha*src1
        scal.submit<std::tuple<T>>(nelems, alpha, src1, dst);
        return;
    }
    // If beta is non-zero and alpha is zero then reduce to scal
    if(alpha == zero)
    {
        // dst = beta*src2
        scal.submit<std::tuple<T>>(nelems, beta, src2, dst);
        return;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = sizeof(T) * 3 * nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src1.get(),
            STARPU_R, src2.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_W, dst.get(),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Add<std::tuple<nntile::fp64_t>>;
template class Add<std::tuple<nntile::fp32_t>>;
template class Add<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Add<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Add<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Add<std::tuple<nntile::bf16_t>>;

//! Pack of add operations for different types
add_pack_t add;

} // namespace nntile::starpu
