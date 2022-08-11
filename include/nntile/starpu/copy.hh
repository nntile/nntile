/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/copy.hh
 * Smart copy StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-11
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/starpu.hh>

namespace nntile
{
namespace starpu
{

//! Smart copying through StarPU buffers
template<typename T>
void copy_cpu(void *buffers[], void *cl_args)
    noexcept;

extern StarpuCodelet copy_codelet_fp32, copy_codelet_fp64;

void copy_restrict_where(uint32_t where);

void copy_restore_where();

template<typename T>
void copy(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape,
        starpu_data_handle_t src, starpu_data_handle_t dst,
        starpu_data_handle_t tmp_index, starpu_data_access_mode mode);

} // namespace starpu
} // namespace nntile

