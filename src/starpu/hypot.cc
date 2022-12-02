/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
 * @date 2022-12-02
 * */

#include "nntile/starpu/hypot.hh"
#include "nntile/kernel/hypot.hh"

namespace nntile
{
namespace starpu
{
namespace hypot
{

//! Complex copying through StarPU buffers is available only on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::hypot::cpu<T>(src, dst);
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
void submit(Handle src, Handle dst)
{
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in hypot task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Handle src, Handle dst);

template
void submit<fp64_t>(Handle src, Handle dst);

} // namespace hypot
} // namespace starpu
} // namespace nntile

