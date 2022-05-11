/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile
{

//! Asynchronous tile-wise sum and scaled sum of squares along given axes
//
// @param[in] src: Source tile to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axes
// @param[in] axes: Axes to be used
//
// For example, if src is a 4-by-5-by-6 tile and axes contains two values 0
// and 2, then output sum_sumssq is 2-dimensional tile of shape (3,5), and
// sum_sumssq[0,i] is an average value, sum_sumssq[1,i] is a maximum absolute
// value and sum_ssq[2,i] is a scaled sum of squares over slice src[:,i,:].
// If src is again a 4-by-5-by-6 tile and axes contains one value 1, then
// output sum_sumssq is 3-dimensional tile of shape (3,4,6), and
// sum_sumssq[0,i,j] is an average value, sum_sumssq[1,i,j] is a maximum
// absolute value and sum_ssq[2,i,j] is a scaled sum of squares over slice
// src[i,:,j].
template<typename T>
void norm_sum_ssq_async(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true);

extern template
void norm_sum_ssq_async(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true);

extern template
void norm_sum_ssq_async(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true);

//! Blocking tile-wise sum and scaled sum of squares along given axes
//
// @param[in] src: Source tile to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axes
// @param[in] axes: Axes to be used
//
// For example, if src is a 4-by-5-by-6 tile and axes contains two values 0
// and 2, then output sum_sumssq is 2-dimensional tile of shape (3,5), and
// sum_sumssq[0,i] is an average value, sum_sumssq[1,i] is a maximum absolute
// value and sum_ssq[2,i] is a scaled sum of squares over slice src[:,i,:].
// If src is again a 4-by-5-by-6 tile and axes contains one value 1, then
// output sum_sumssq is 3-dimensional tile of shape (3,4,6), and
// sum_sumssq[0,i,j] is an average value, sum_sumssq[1,i,j] is a maximum
// absolute value and sum_ssq[2,i,j] is a scaled sum of squares over slice
// src[i,:,j].
template<typename T>
void norm_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true)
{
    norm_sum_ssq_async(src, sum_ssq, axes, init_output);
    starpu_task_wait_for_all();
}

template<typename T>
void norm_sum_ssq_async(const Tile<T> &src, const Tile<T> &sum_ssq,
        Index axis, bool init_output=true);

extern template
void norm_sum_ssq_async(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        Index axis, bool init_output=true);

extern template
void norm_sum_ssq_async(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        Index axis, bool init_output=true);

template<typename T>
void norm_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis,
        bool init_output=true)
{
    norm_sum_ssq_async(src, sum_ssq, axis, init_output);
    starpu_task_wait_for_all();
}

template<typename T>
void norm_avg_dev_async(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps);

extern template
void norm_avg_dev_async(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &avg_dev, Index nelems, fp32_t eps);

extern template
void norm_avg_dev_async(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &avg_dev, Index nelems, fp64_t eps);

template<typename T>
void norm_avg_dev(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev_async(sum_ssq, avg_dev, nelems, eps);
    starpu_task_wait_for_all();
}

} // namespace nntile

