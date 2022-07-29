/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/clear.cc
 * Clear Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/clear.hh"
#include "nntile/kernel/cpu/clear.hh"

namespace nntile
{

starpu_perfmodel clear_perfmodel =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_clear",
};

StarpuCodelet clear_codelet("nntile_clear",
        &clear_perfmodel,
        {clear_starpu_cpu},
        {}
        );

template<typename T>
void clear_work(const Tile<T> &src)
{
    int ret = starpu_task_insert(&clear_codelet,
            STARPU_W, static_cast<starpu_data_handle_t>(src),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation
template
void clear_work<fp32_t>(const Tile<fp32_t> &src);

template
void clear_work<fp64_t>(const Tile<fp64_t> &src);

} // namespace nntile

