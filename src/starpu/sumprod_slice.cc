/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/sumprod_slice.cc
 * Sums over fibers into a slice of a product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/sumprod_slice.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/sumprod_slice.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
SumProdSlice<std::tuple<T>>::SumProdSlice():
    codelet("nntile_sumprod_slice", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::sumprod_slice::cpu<T>
template<typename T>
void SumProdSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args) noexcept
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
    kernel::sumprod_slice::cpu<T>(args->m, args->n, args->k, args->alpha, src1,
            src2, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void SumProdSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::sumprod_slice::cuda<T>
template<typename T>
void SumProdSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args) noexcept
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
    kernel::sumprod_slice::cuda<T>(stream, args->m, args->n, args->k,
            args->alpha, src1, src2, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SumProdSlice<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SumProdSlice<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SumProdSlice<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for sumprod_slice tasks
template<typename T>
uint32_t SumProdSlice<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void SumProdSlice<std::tuple<T>>::submit(Index m, Index n, Index k, Scalar alpha, Handle src1, Handle src2,
        Scalar beta, Handle dst, int redux)
//! Insert sumprod_slice task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr Scalar zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        if(redux != 0)
        {
            dst_mode = STARPU_REDUX;
        }
        else
        {
            dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
        }
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    size_t src1_nbytes = sizeof(T) * m * k * n;
    size_t dst_nbytes = sizeof(T) * m * n;
    double nflops = beta == 0.0 ? 2*src1_nbytes + dst_nbytes :
        2 * (src1_nbytes+dst_nbytes);
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src1.get(),
            STARPU_R, src2.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, dst.get(),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in sumprod_slice task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SumProdSlice<std::tuple<nntile::fp64_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SumProdSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SumProdSlice<std::tuple<nntile::bf16_t>>;

//! Pack of sumprod_slice operations for different types
sumprod_slice_pack_t sumprod_slice;

} // namespace nntile::starpu
