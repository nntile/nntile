/*! @randnright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/randn.hh
 * Randn operation on StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{

// Randn operation on StarPU buffers
template<typename T>
void randn_cpu(void *buffers[], void *cl_args)
    noexcept;

extern StarpuCodelet randn_codelet_fp32, randn_codelet_fp64;

void randn_init();

void randn_restrict_where(uint32_t where);

void randn_restore_where();

template<typename T>
void randn(Index ndim, Index nelems, unsigned long long seed,
        T mean, T stddev, const std::vector<Index> &start,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        const std::vector<Index> &underlying_shape, starpu_data_handle_t data,
        starpu_data_handle_t tmp_index);

} // namespace starpu
} // namespace nntile

