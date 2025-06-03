/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/mask_scalar.cc
 * StarPU wrappers for mask_scalar operation
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/mask_scalar.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/mask_scalar.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
MaskScalar<std::tuple<T>>::MaskScalar():
    codelet("nntile_mask_scalar", footprint, cpu_funcs, cuda_funcs)
{
}

//! Mask scalar operation for StarPU buffer on CPU
template<typename T>
void MaskScalar<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    const bool_t* mask = interfaces[1]->get_ptr<bool_t>();
    // Launch kernel
    kernel::mask_scalar::cpu<T>(args->nrows, args->ncols, mask, args->val,
            data);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Mask scalar StarPU buffer on CUDA
template<typename T>
void MaskScalar<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    const bool_t *mask = interfaces[1]->get_ptr<bool_t>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::mask_scalar::cuda<T>(stream, args->nrows, args->ncols, mask,
            args->val, data);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for mask_scalar tasks
template<typename T>
uint32_t MaskScalar<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nrows, sizeof(args->nrows), hash);
    hash = starpu_hash_crc32c_be_n(&args->ncols, sizeof(args->ncols), hash);
    return hash;
}

template<typename T>
void MaskScalar<std::tuple<T>>::submit(
        Index nrows, Index ncols, Handle mask, Scalar val, Handle data)
//! Insert mask_scalar task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nrows = nrows;
    args->ncols = ncols;
    args->val = val;
    // Indicate maximal possible amount of writes as flops count
    double nflops = sizeof(T) * nrows * (ncols+1);
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, data.get(),
            STARPU_R, mask.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in mask_scalar task submission");
    }
}

//! Pack of mask_scalar operations for different types
mask_scalar_pack_t mask_scalar;

} // namespace nntile::starpu
