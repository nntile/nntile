/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add.cc
 * Add operation on a StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-21
 * */

#include "nntile/starpu/add.hh"
#include "nntile/kernel/add.hh"
#include <cstdlib>

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for add operation
namespace add
{

//! Apply add operation for StarPU buffers in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::add::cpu<T>(args->num_elements, args->alpha, src, args->beta, dst);
}

#ifdef NNTILE_USE_CUDA
//! Apply add for StarPU buffers on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::add::cuda<T>(stream, args->num_elements, args->alpha, src,
            args->beta, dst);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_add_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_add_fp64",
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
void submit(Index num_elements, T alpha, Handle src, T beta, Handle dst)
//! Insert add task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Access mode for the dst handle
    constexpr T zero = 0, one = 1;
    enum starpu_data_access_mode dst_mode;
    if(beta == zero)
    {
        dst_mode = STARPU_W;
    }
    else if(beta == one)
    {
        dst_mode = Config::STARPU_RW_COMMUTE;
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t<T> *args = (args_t<T> *)std::malloc(sizeof(*args));
    args->num_elements = num_elements;
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, static_cast<starpu_data_handle_t>(dst), 0);
            // STARPU_FLOPS, nflops);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index num_elements, fp32_t alpha, Handle src, fp32_t beta,
        Handle dst);

template
void submit<fp64_t>(Index num_elements, fp64_t alpha, Handle src, fp64_t beta,
        Handle dst);

} // namespace add
} // namespace starpu
} // namespace nntile

