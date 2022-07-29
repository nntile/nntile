/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/scal.cc
 * Scaling operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/scal.hh"
#include "nntile/kernel/cpu/scal.hh"

namespace nntile
{

starpu_perfmodel scal_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_scal_fp32",
};

starpu_perfmodel scal_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_scal_fp64",
};

StarpuCodelet scal_codelet_fp32("nntile_scal_fp32",
        &scal_perfmodel_fp32,
        {scal_starpu_cpu<fp32_t>},
        {}
        );

StarpuCodelet scal_codelet_fp64("nntile_scal_fp64",
        &scal_perfmodel_fp64,
        {scal_starpu_cpu<fp64_t>},
        {}
        );

template<typename T>
void scal_work(const Tile<T> &src, T alpha)
{
    int ret = starpu_task_insert(scal_codelet<T>(),
            STARPU_VALUE, &src.nelems, sizeof(src.nelems),
            STARPU_VALUE, &alpha, sizeof(alpha),
            STARPU_RW, static_cast<starpu_data_handle_t>(src),
            STARPU_FLOPS, static_cast<double>(src.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

template
void scal_work<fp32_t>(const Tile<fp32_t> &src, fp32_t alpha);

template
void scal_work<fp64_t>(const Tile<fp64_t> &src, fp64_t alpha);

} // namespace nntile

