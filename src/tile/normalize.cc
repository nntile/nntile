/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/normalize.cc
 * Normalize operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/normalize.hh"
#include "nntile/kernel/cpu/normalize.hh"

namespace nntile
{

starpu_perfmodel normalize_perfmodel_fp32 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_normalize_fp32",
};

starpu_perfmodel normalize_perfmodel_fp64 =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "nntile_normalize_fp64",
};

StarpuCodelet normalize_codelet_fp32("nntile_normalize_fp32",
        &normalize_perfmodel_fp32,
        {normalize_starpu_cpu<fp32_t>},
        {}
        );

StarpuCodelet normalize_codelet_fp64("nntile_normalize_fp64",
        &normalize_perfmodel_fp64,
        {normalize_starpu_cpu<fp64_t>},
        {}
        );

// Normalization operation over single axis
template<typename T>
void normalize_work(const StarpuVariableHandle &gamma_beta,
        const Tile<T> &sumnorm, const Tile<T> &dst, Index l, T eps,
        Index axis)
{
    // Reshape inputs for simplicity: sumnorm -> (2,m,n), dst -> (m,k,n)
    // dst is a part of (m,l,n) tensor
    Index m, n, k;
    if(axis == 0)
    {
        m = 1;
        n = sumnorm.nelems / 2; // 2 elements per single n
        k = dst.shape[0];
    }
    else if(axis == dst.ndim-1)
    {
        m = sumnorm.nelems / 2; // 2 elements per single m
        n = 1;
        k = dst.shape[axis];
    }
    else
    {
        m = dst.stride[axis];
        n = dst.matrix_shape[axis+1][1];
        k = dst.shape[axis];
    }
    // Insert corresponding task
    int ret = starpu_task_insert(normalize_codelet<T>(),
            STARPU_VALUE, &m, sizeof(m),
            STARPU_VALUE, &n, sizeof(n),
            STARPU_VALUE, &k, sizeof(k),
            STARPU_VALUE, &l, sizeof(l),
            STARPU_VALUE, &eps, sizeof(eps),
            STARPU_R, static_cast<starpu_data_handle_t>(gamma_beta),
            STARPU_R, static_cast<starpu_data_handle_t>(sumnorm),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            STARPU_FLOPS, static_cast<double>(14*dst.nelems),
            0);
    if(ret != 0)
    {
        throw std::runtime_error("ret != 0");
    }
}

// Explicit instantiation
template
void normalize_work<fp32_t>(const StarpuVariableHandle &gamma_beta,
        const Tile<fp32_t> &sumnorm, const Tile<fp32_t> &dst, Index l,
        fp32_t eps, Index axis);

template
void normalize_work<fp64_t>(const StarpuVariableHandle &gamma_beta,
        const Tile<fp64_t> &sumnorm, const Tile<fp64_t> &dst, Index l,
        fp64_t eps, Index axis);

} // namespace nntile

