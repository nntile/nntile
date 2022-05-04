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
// @param[in] axis: Dimension index of the bias
//
// The source tile shall have 1 dimension less than the destination tile,
// as this operation does the following update:
// dst[i_0, ..., i_b-1, i_b, i_b+1, ..., i_d-1] += src[i_0, ..., i_b-1, i_b+1,
// ..., i_d-1]
// where b is the axis and i_d is the src.ndim
template<typename T>
void bias_async(const Tile<T> &src, const Tile<T> &dst, Index axis);

extern template
void bias_async(const Tile<fp32_t> &src, const Tile<fp32_t> &dst,
        Index axis);

extern template
void bias_async(const Tile<fp64_t> &src, const Tile<fp64_t> &dst,
        Index axis);

//! Blocking version of tile-wise bias operation
//
// @param[in] src: Source of the bias
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
//
// The source tile shall have 1 dimension less than the destination tile,
// as this operation does the following update:
// dst[i_0, ..., i_b-1, i_b, i_b+1, ..., i_d-1] += src[i_0, ..., i_b-1, i_b+1,
// ..., i_d-1]
// where b is the axis and i_d is the src.ndim
template<typename T>
void bias(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    bias_async<T>(src, dst, axis);
    starpu_task_wait_for_all();
}

//! Asynchronous tile-wise bias operation by averages and deviations
//
// @param[in] avg_dev: Source of the bias (averages and deviations)
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
template<typename T>
void bias_avg_dev_async(const Tile<T> &avg_dev, const Tile<T> &dst,
        Index axis);

extern template
void bias_avg_dev_async(const Tile<fp32_t> &avg_dev, const Tile<fp32_t> &dst,
        Index axis);

extern template
void bias_avg_dev_async(const Tile<fp64_t> &avg_dev, const Tile<fp64_t> &dst,
        Index axis);

//! Blocking version of tile-wise bias operation by averages and deviations
//
// @param[in] avg_dev: Source of the bias (averages and deviations)
// @param[inout] dst: Destination of the bias
// @param[in] axis: Dimension index of the bias
template<typename T>
void bias_avg_dev(const Tile<T> &avg_dev, const Tile<T> &dst, Index axis)
{
    bias_avg_dev_async<T>(avg_dev, dst, axis);
    starpu_task_wait_for_all();
}

} // namespace nntile

