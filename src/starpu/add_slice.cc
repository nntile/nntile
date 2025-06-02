/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_slice.cc
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/add_slice.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_slice.hh"
#include "nntile/starpu/add.hh"

//! StarPU wrappers for add_slice operation
namespace nntile::starpu
{

//! Constructor
template<typename T>
AddSlice<std::tuple<T>>::AddSlice():
    codelet("nntile_add_slice", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::add_slice::cpu<T>
template<typename T>
void AddSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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
    kernel::add_slice::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->alpha,
        src1,
        args->beta,
        src2,
        dst
    );
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::add_slice::cuda<T>
template<typename T>
void AddSlice<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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
    kernel::add_slice::cuda<T>(
        stream,
        args->m,
        args->n,
        args->k,
        args->alpha,
        src1,
        args->beta,
        src2,
        dst
    );
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for add_slice tasks
template<typename T>
uint32_t AddSlice<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->m, sizeof(args->m), hash);
    hash = starpu_hash_crc32c_be_n(&args->n, sizeof(args->n), hash);
    hash = starpu_hash_crc32c_be_n(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void AddSlice<std::tuple<T>>::submit(
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    Handle src1,
    Scalar beta,
    Handle src2,
    Handle dst
)
{
    constexpr Scalar zero = 0.0, one = 1.0;
    // If k is 1, then this operation reduces to add
    if(k == 1)
    {
        add.submit<T>(m*n, alpha, src1, beta, src2, dst);
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
        dst_mode = STARPU_RW | STARPU_COMMUTE;
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
    double nflops = m * n * (2*k+1);
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
        throw std::runtime_error("Error in add_slice task submission");
    }
}

//! Pack of add_slice operations for different types
add_slice_pack_t add_slice;

} // namespace nntile::starpu
