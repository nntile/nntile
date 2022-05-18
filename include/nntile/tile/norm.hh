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

//! Accumulate sum and scaled sum of squares of 2 tiles
//
// Main computational routine that does NO argument checking.
// @param[in] sum_ssq: Sum and scaled sum of squares of some tile
// @param[inout] sum_ssq_total: Sum and scaled sum of squares of another
//      tile. On output, contains accumulated values.
template<typename T>
void norm_sum_ssq_accumulate_work(const Tile<T> &sum_ssq,
        const Tile<T> &sum_ssq_total);

extern template
void norm_sum_ssq_accumulate_work(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &sum_ssq_total);

extern template
void norm_sum_ssq_accumulate_work(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &sum_ssq_total);

//! Accumulate sum and scaled sum of squares of 2 tiles
//
// Checks input arguments
template<typename T>
void norm_sum_ssq_accumulate_async(const Tile<T> &sum_ssq,
        const Tile<T> &sum_ssq_total)
{
    // Check dimensions
    if(sum_ssq.ndim != sum_ssq_total.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != sum_ssq_total.ndim");
    }
    // Check shapes
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(sum_ssq_total.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq_total.shape[0] != 3");
    }
    for(Index i = 1; i < sum_ssq.ndim; ++i)
    {
        if(sum_ssq.shape[i] != sum_ssq_total.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != "
                    "sum_ssq_total.shape[i]");
        }
    }
    // Launch codelet
    norm_sum_ssq_accumulate_work(sum_ssq, sum_ssq_total);
}

//! Accumulate sum and scaled sum of squares of 2 tiles
//
// Checks input arguments and blocks until finished
template<typename T>
void norm_sum_ssq_accumulate(const Tile<T> &sum_ssq,
        const Tile<T> &sum_ssq_total)
{
    norm_sum_ssq_accumulate_async(sum_ssq, sum_ssq_total);
    starpu_task_wait_for_all();
}

//! Tile-wise sum and scaled sum of squares along given set of axes
//
// Main computational routine that does NO argument checking.
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
void norm_sum_ssq_work(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, const StarpuVariableHandle &scratch,
        bool init_output=true);

extern template
void norm_sum_ssq_work(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        const std::vector<Index> &axes, const StarpuVariableHandle &scratch,
        bool init_output=true);

extern template
void norm_sum_ssq_work(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        const std::vector<Index> &axes, const StarpuVariableHandle &scratch,
        bool init_output=true);

//! Tile-wise sum and scaled sum of squares along given set of axes
//
// Checks input arguments
template<typename T>
void norm_sum_ssq_async(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true)
{
    // Check dimensions
    if(src.ndim+1 != sum_ssq.ndim+axes.size())
    {
        throw std::runtime_error("src.ndim+1 != sum_ssq.ndim+axes.size()");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Treat special case of empty axes
    if(axes.size() == 0)
    {
        throw std::runtime_error("Empty axes");
    }
    // Check axes
    if(axes[0] < 0)
    {
        throw std::runtime_error("axes[0] < 0");
    }
    if(axes[axes.size()-1] >= src.ndim)
    {
        throw std::runtime_error("axes[axes.size()-1] >= src.ndim");
    }
    for(Index i = 1; i < axes.size(); ++i)
    {
        if(axes[i] <= axes[i-1])
        {
            throw std::runtime_error("axes[i] <= axes[i-1]");
        }
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    for(Index j = 0; j < axes[0]; ++j)
    {
        if(src.shape[j] != sum_ssq.shape[j+1])
        {
            throw std::runtime_error("src.shape[j] != sum_ssq.shape[j+1]");
        }
    }
    for(Index i = 1; i < axes.size(); ++i)
    {
        for(Index j = axes[i-1]+1; j < axes[i]; ++j)
        {
            if(src.shape[j] != sum_ssq.shape[j+1-i])
            {
                throw std::runtime_error("src.shape[j] != "
                        "sum_ssq.shape[j+1-i]");
            }
        }
    }
    for(Index j = axes[axes.size()-1]+1; j < src.ndim; ++j)
    {
        if(src.shape[j] != sum_ssq.shape[j+1-axes.size()])
        {
            throw std::runtime_error("src.shape[j] != "
                    "sum_ssq.shape[j+1-axes.size()]");
        }
    }
    // Scratch buffer
    StarpuVariableHandle scratch(2 * (src.ndim+sum_ssq.ndim) * sizeof(Index));
    // Launch codelet
    norm_sum_ssq_work(src, sum_ssq, axes, scratch, init_output);
}

//! Tile-wise sum and scaled sum of squares along given set of axes
//
// Checks input arguments and blocks until finished
template<typename T>
void norm_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes, bool init_output=true)
{
    norm_sum_ssq_async(src, sum_ssq, axes, init_output);
    starpu_task_wait_for_all();
}

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Main computational routine that does NO argument checking.
// @param[in] src: Source tile to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axes
// @param[in] axes: Axes to be used
template<typename T>
void norm_sum_ssq_work(const Tile<T> &src, const Tile<T> &sum_ssq,
        Index axis, bool init_output=true);

extern template
void norm_sum_ssq_work(const Tile<fp32_t> &src, const Tile<fp32_t> &sum_ssq,
        Index axis, bool init_output=true);

extern template
void norm_sum_ssq_work(const Tile<fp64_t> &src, const Tile<fp64_t> &sum_ssq,
        Index axis, bool init_output=true);

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Checks input arguments
template<typename T>
void norm_sum_ssq_async(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis,
        bool init_output=true)
{
    // Check dimensions
    if(src.ndim != sum_ssq.ndim)
    {
        throw std::runtime_error("src.ndim != sum_ssq.ndim");
    }
    // Treat special case of src.ndim=0
    if(src.ndim == 0)
    {
        throw std::runtime_error("Scalar input makes no sense");
    }
    // Check axis
    if(axis < 0)
    {
        throw std::runtime_error("axis < 0");
    }
    if(axis >= src.ndim)
    {
        throw std::runtime_error("axis >= src.ndim");
    }
    // Check shapes of src and sum_ssq
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    for(Index i = 0; i < axis; ++i)
    {
        if(src.shape[i] != sum_ssq.shape[i+1])
        {
            throw std::runtime_error("src.shape[i] != sum_ssq.shape[i+1]");
        }
    }
    for(Index i = axis+1; i < src.ndim; ++i)
    {
        if(src.shape[i] != sum_ssq.shape[i])
        {
            throw std::runtime_error("src.shape[i] != sum_ssq.shape[i]");
        }
    }
    // Launch codelet
    norm_sum_ssq_work(src, sum_ssq, axis, init_output);
}

//! Tile-wise sum and scaled sum of squares along single given axis
//
// Checks input arguments and blocks until finished
template<typename T>
void norm_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis,
        bool init_output=true)
{
    norm_sum_ssq_async(src, sum_ssq, axis, init_output);
    starpu_task_wait_for_all();
}

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Main computational routine that does NO argument checking.
template<typename T>
void norm_avg_dev_work(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps);

extern template
void norm_avg_dev_work(const Tile<fp32_t> &sum_ssq,
        const Tile<fp32_t> &avg_dev, Index nelems, fp32_t eps);

extern template
void norm_avg_dev_work(const Tile<fp64_t> &sum_ssq,
        const Tile<fp64_t> &avg_dev, Index nelems, fp64_t eps);

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Checks input arguments
template<typename T>
void norm_avg_dev_async(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    // Check inputs
    if(sum_ssq.ndim != avg_dev.ndim)
    {
        throw std::runtime_error("sum_ssq.ndim != avg_dev.ndim");
    }
    // Input shape dimension shall be at least 1
    if(sum_ssq.ndim == 0)
    {
        throw std::runtime_error("sum_ssq.ndim == 0");
    }
    // Check number of elements
    if(nelems <= 0)
    {
        throw std::runtime_error("nelems <= 0");
    }
    // Check regularization
    if(eps < 0)
    {
        throw std::runtime_error("eps < 0");
    }
    // Check shapes
    if(sum_ssq.shape[0] != 3)
    {
        throw std::runtime_error("sum_ssq.shape[0] != 3");
    }
    if(avg_dev.shape[0] != 2)
    {
        throw std::runtime_error("avg_dev.shape[0] != 2");
    }
    for(Index i = 1; i < sum_ssq.ndim; ++i)
    {
        if(sum_ssq.shape[i] != avg_dev.shape[i])
        {
            throw std::runtime_error("sum_ssq.shape[i] != avg_dev.shape[i]");
        }
    }
    // Launch codelet
    norm_avg_dev_work<T>(sum_ssq, avg_dev, nelems, eps);
}

//! Tile-wise average and deviation from sum and scaled sum of squares
//
// Checks input arguments and blocks until finished
template<typename T>
void norm_avg_dev(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev_async(sum_ssq, avg_dev, nelems, eps);
    starpu_task_wait_for_all();
}

} // namespace nntile

