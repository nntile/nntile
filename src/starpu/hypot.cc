/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot.cc
 * Hypot on two StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/starpu/hypot.hh"
#include "nntile/kernel/hypot.hh"

namespace nntile
{
namespace starpu
{
namespace hypot
{

//! Hypothenus calculation
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *x = interfaces[0]->get_ptr<T>();
    T *y = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::hypot::cpu<T>(args->alpha, x, args->beta, y);
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_hypot_fp32",
            nullptr,
            {cpu<fp32_t>},
            {}
            );
    codelet_fp64.init("nntile_hypot_fp64",
            nullptr,
            {cpu<fp64_t>},
            {}
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
void submit(T alpha, Handle src, T beta, Handle dst)
//! Insert hypot task into StarPU pool of tasks
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
    args->alpha = alpha;
    args->beta = beta;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            dst_mode, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in hypot task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(fp32_t alpha, Handle src, fp32_t beta, Handle dst);

template
void submit<fp64_t>(fp64_t alpha, Handle src, fp64_t beta, Handle dst);

} // namespace hypot
} // namespace starpu
} // namespace nntile

