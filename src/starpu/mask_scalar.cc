/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/mask_scalar.cc
 * Mask scalar operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-29
 * */

#include "nntile/starpu/mask_scalar.hh"
#include "nntile/kernel/mask_scalar.hh"

namespace nntile
{
namespace starpu
{
namespace mask_scalar
{

//! Mask scalar operation for StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    const bool_t* mask = interfaces[1]->get_ptr<bool_t>();
    // Launch kernel
    kernel::mask_scalar::cpu<T>(args->nrows, args->ncols, mask, args->val,
            data);
}

#ifdef NNTILE_USE_CUDA
//! Mask scalar StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    const bool_t *mask = interfaces[1]->get_ptr<bool_t>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::mask_scalar::cuda<T>(stream, args->nrows, args->ncols, mask,
            args->val, data);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_mask_scalar_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_mask_scalar_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index nrows, Index ncols, Handle mask, T val, Handle data)
//! Insert mask_scalar task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->nrows = nrows;
    args->ncols = ncols;
    args->val = val;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(data),
            STARPU_R, static_cast<starpu_data_handle_t>(mask),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in mask_scalar task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Index nrows, Index ncols, Handle mask, fp32_t val,
        Handle data);

template
void submit<fp64_t>(Index nrows, Index ncols, Handle mask, fp64_t val,
        Handle data);

} // namespace mask_scalar
} // namespace starpu
} // namespace nntile

