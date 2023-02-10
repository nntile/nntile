/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_scalar.cc
 * Add scalar to StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/starpu/add_scalar.hh"
#include "nntile/kernel/add_scalar.hh"

namespace nntile
{
namespace starpu
{
namespace add_scalar
{

//! Complex copying through StarPU buffers is available only on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *src = interfaces[0]->get_ptr<T>();
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Launch kernel
    kernel::add_scalar::cpu<T>(args->val, args->nelems, src);
}

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_add_scalar_fp32",
            nullptr,
            {cpu<fp32_t>},
            {}
            );
    codelet_fp64.init("nntile_add_scalar_fp64",
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
void submit(T val, Index num_elements, Handle src)
{
    // Submit task
    auto cl_args = new args_t<T>{val, num_elements};
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, cl_args, sizeof(*cl_args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in add_scalar task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(fp32_t val, Index num_elements, Handle src);

template
void submit<fp64_t>(fp64_t val, Index num_elements, Handle src);

} // namespace add_scalar
} // namespace starpu
} // namespace nntile

