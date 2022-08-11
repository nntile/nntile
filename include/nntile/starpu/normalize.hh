/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/normalize.hh
 * Normalize operation for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-10
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{

//! Structure for arguments
template<typename T>
struct normalize_args
{
    Index m;
    Index n;
    Index k;
    Index l;
    T eps;
};

// Apply normalize along middle axis of StarPU buffer on CPU
template<typename T>
void normalize_cpu(void *buffers[], void *cl_args)
    noexcept;

extern starpu_perfmodel normalize_perfmodel_fp32, normalize_perfmodel_fp64;

extern StarpuCodelet normalize_codelet_fp32, normalize_codelet_fp64;

void normalize_restrict_where(uint32_t where);

void normalize_restore_where();

template<typename T>
void normalize(Index m, Index n, Index k, Index l, T eps,
        starpu_data_handle_t gamma_beta, starpu_data_handle_t src,
        starpu_data_handle_t dst);

} // namespace starpu
} // namespace nntile

