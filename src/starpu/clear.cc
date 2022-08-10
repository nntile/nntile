/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/clear.cc
 * Clear a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-05
 * */

#include "nntile/starpu/clear.hh"
#include <cstring>

namespace nntile
{
namespace starpu
{

// Clear a StarPU buffer on CPU
void clear_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // No arguments
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    std::size_t size = interfaces[0]->elemsize;
    void *data = interfaces[0]->get_ptr<void>();
    std::memset(data, 0, size);
}

// No custom footprint as buffer size is enough for this purpose
starpu_perfmodel clear_perfmodel =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_clear",
};

StarpuCodelet clear_codelet("nntile_clear",
        &clear_perfmodel,
        {clear_cpu},
        {}
        );

void clear_restrict_where(uint32_t where)
{
    clear_codelet.restrict_where(where);
}

void clear_restore_where()
{
    clear_codelet.restore_where();
}

//! Insert task to clear buffer
void clear(starpu_data_handle_t data)
{
    // Submit task
    int ret = starpu_task_insert(&clear_codelet,
            STARPU_W, data,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in clear task submission");
    }
}

} // namespace starpu
} // namespace nntile

