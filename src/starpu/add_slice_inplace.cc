/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_slice_inplace.cc
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/add_slice_inplace.hh"

// Standard libraries
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/add_slice_inplace.hh"
#include "nntile/starpu/add_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AddSliceInplace<std::tuple<T>>::AddSliceInplace():
    codelet("nntile_add_slice_inplace", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::add_slice_inplace::cpu<T>
template<typename T>
void AddSliceInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::add_slice_inplace::cpu<T>(
        args->m, args->n, args->k, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddSliceInplace<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::add_slice_inplace::cuda<T>
template<typename T>
void AddSliceInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::add_slice_inplace::cuda<T>(
        stream, args->m, args->n, args->k, args->alpha, src, args->beta, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddSliceInplace<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AddSliceInplace<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSliceInplace<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_slice_inplace tasks
template<typename T>
uint32_t AddSliceInplace<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void AddSliceInplace<std::tuple<T>>::submit(
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    Handle src,
    Scalar beta,
    Handle dst
)
//! Insert add_slice_inplace task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    constexpr Scalar zero = 0.0, one = 1.0;
    // If k is 1, then this operation reduces to add
    if(k == 1)
    {
        add_inplace.submit<std::tuple<T>>(m*n, alpha, src, beta, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    // Put amount of bytes read and write inplace of gflops
    double nflops = beta == zero ? sizeof(T)*m*(k+1)*n :
            sizeof(T)*m*(2*k+1)*n;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, dst.get(),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add_slice_inplace task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddSliceInplace<std::tuple<nntile::fp64_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddSliceInplace<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddSliceInplace<std::tuple<nntile::bf16_t>>;

//! Pack of add_slice_inplace operations for different types
add_slice_inplace_pack_t add_slice_inplace;

} // namespace nntile::starpu
