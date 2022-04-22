/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/bias.hh
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

//! Asynchronous tile-wise bias operation
//
// @param[in] src: Source of the bias
// @param[inout] dst: Destination of the bias
// @param[in] batch_dim: Dimension index of the bias
//
// The source tile shall have 1 dimension less than the destination tile,
// as this operation does the following update:
// dst[i_0, ..., i_b-1, i_b, i_b+1, ..., i_d-1] += src[i_0, ..., i_b-1, i_b+1,
// ..., i_d-1]
// where b is the batch_dim and i_d is the src.ndim
template<typename T>
void bias_async(const Tile<T> &src, const Tile<T> &dst, Index batch_dim);

extern template
void bias_async(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index batch_dim);

extern template
void bias_async(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index batch_dim);

//! Blocking version of tile-wise bias operation
//
// @param[in] src: Source of the bias
// @param[inout] dst: Destination of the bias
// @param[in] batch_dim: Dimension index of the bias
//
// The source tile shall have 1 dimension less than the destination tile,
// as this operation does the following update:
// dst[i_0, ..., i_b-1, i_b, i_b+1, ..., i_d-1] += src[i_0, ..., i_b-1, i_b+1,
// ..., i_d-1]
// where b is the batch_dim and i_d is the src.ndim
template<typename T>
void bias(const Tile<T> &src, const Tile<T> &dst, Index batch_dim)
{
    bias_async<T>(src, dst, batch_dim);
    starpu_task_wait_for_all();
}

} // namespace nntile

