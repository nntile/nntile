/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gelu.cc
 * GeLU operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/gelu.hh"
#include "nntile/kernel/cpu/gelu.hh"

namespace nntile
{

starpu_perfmodel gelu_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_gelu_fp32",
};

starpu_perfmodel gelu_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_gelu_fp64",
};

StarpuCodelet gelu_codelet_fp32("nntile_gelu_fp32",
        &gelu_perfmodel_fp32,
        {gelu_starpu_cpu<fp32_t>},
        {}
        );

StarpuCodelet gelu_codelet_fp64("nntile_gelu_fp64",
        &gelu_perfmodel_fp64,
        {gelu_starpu_cpu<fp64_t>},
        {}
        );

template<typename T>
void gelu_work(const Tile<T> &A)
{
    int ret = starpu_task_insert(gelu_codelet<T>(),
            STARPU_VALUE, &A.nelems, sizeof(A.nelems),
            STARPU_RW, static_cast<starpu_data_handle_t>(A),
            // std::erf is assumed as a single flop
            STARPU_FLOPS, static_cast<double>(5*A.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

template
void gelu_work(const Tile<fp32_t> &A);

template
void gelu_work(const Tile<fp64_t> &A);

} // namespace nntile

