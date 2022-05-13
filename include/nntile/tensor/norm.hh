/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/norm.hh
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Asynchronously accumulate sum and scaled sum of squares
//
// @param[in] sum_ssq: Sum and scaled sum of squares of some tensor
// @param[inout] sum_ssq_total: Sum and scaled sum of squares of another
//      tensor. On output, contains accumulated values.
template<typename T>
void norm_sum_ssq_accumulate_async(const Tensor<T> &sum_ssq,
        const Tensor<T> &sum_ssq_total);

extern template
void norm_sum_ssq_accumulate_async(const Tensor<fp32_t> &sum_ssq,
        const Tensor<fp32_t> &sum_ssq_total);

extern template
void norm_sum_ssq_accumulate_async(const Tensor<fp64_t> &sum_ssq,
        const Tensor<fp64_t> &sum_ssq_total);

//! Blocking version of accumulate sum and scaled sum of squares
//
// @param[in] sum_ssq: Sum and scaled sum of squares of some tensor
// @param[inout] sum_ssq_total: Sum and scaled sum of squares of another
//      tensor. On output, contains accumulated values.
template<typename T>
void norm_sum_ssq_accumulate(const Tensor<T> &sum_ssq,
        const Tensor<T> &sum_ssq_total)
{
    norm_sum_ssq_accumulate_async(sum_ssq, sum_ssq_total);
    starpu_task_wait_for_all();
}

//! Asynchronous tensor-wise sum and scaled sum of squares along given axes
//
// @param[in] src: Source tensor to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axes
// @param[in] axes: Axes to be used
//
// For example, if src is a 4-by-5-by-6 tensor and axes contains two values 0
// and 2, then output sum_sumssq is 2-dimensional tensor of shape (3,5), and
// sum_sumssq[0,i] is an average value, sum_sumssq[1,i] is a maximum absolute
// value and sum_ssq[2,i] is a scaled sum of squares over slice src[:,i,:].
// If src is again a 4-by-5-by-6 tensor and axes contains one value 1, then
// output sum_sumssq is 3-dimensional tensor of shape (3,4,6), and
// sum_sumssq[0,i,j] is an average value, sum_sumssq[1,i,j] is a maximum
// absolute value and sum_ssq[2,i,j] is a scaled sum of squares over slice
// src[i,:,j].
template<typename T>
void norm_sum_ssq_async(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        const std::vector<Index> &axes);

extern template
void norm_sum_ssq_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sum_ssq, const std::vector<Index> &axes);

extern template
void norm_sum_ssq_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sum_ssq, const std::vector<Index> &axes);

//! Blocking tensor-wise sum and scaled sum of squares along given axes
//
// @param[in] src: Source tensor to get mean and variance
// @param[out] sum_ssq: Sum and scaled sum of squares along given axes
// @param[in] axes: Axes to be used
//
// For example, if src is a 4-by-5-by-6 tensor and axes contains two values 0
// and 2, then output sum_sumssq is 2-dimensional tensor of shape (3,5), and
// sum_sumssq[0,i] is an average value, sum_sumssq[1,i] is a maximum absolute
// value and sum_ssq[2,i] is a scaled sum of squares over slice src[:,i,:].
// If src is again a 4-by-5-by-6 tensor and axes contains one value 1, then
// output sum_sumssq is 3-dimensional tensor of shape (3,4,6), and
// sum_sumssq[0,i,j] is an average value, sum_sumssq[1,i,j] is a maximum
// absolute value and sum_ssq[2,i,j] is a scaled sum of squares over slice
// src[i,:,j].
template<typename T>
void norm_sum_ssq(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        const std::vector<Index> &axes)
{
    norm_sum_ssq_async(src, sum_ssq, axes);
    starpu_task_wait_for_all();
}

template<typename T>
void norm_sum_ssq_async(const Tensor<T> &src, const Tensor<T> &sum_ssq,
        Index axis);

extern template
void norm_sum_ssq_async(const Tensor<fp32_t> &src,
        const Tensor<fp32_t> &sum_ssq, Index axis);

extern template
void norm_sum_ssq_async(const Tensor<fp64_t> &src,
        const Tensor<fp64_t> &sum_ssq, Index axis);

template<typename T>
void norm_sum_ssq(const Tensor<T> &src, const Tensor<T> &sum_ssq, Index axis)
{
    norm_sum_ssq_async(src, sum_ssq, axis);
    starpu_task_wait_for_all();
}

template<typename T>
void norm_avg_dev_async(const Tensor<T> &sum_ssq, const Tensor<T> &avg_dev,
        Index nelems, T eps);

extern template
void norm_avg_dev_async(const Tensor<fp32_t> &sum_ssq,
        const Tensor<fp32_t> &avg_dev, Index nelems, fp32_t eps);

extern template
void norm_avg_dev_async(const Tensor<fp64_t> &sum_ssq,
        const Tensor<fp64_t> &avg_dev, Index nelems, fp64_t eps);

template<typename T>
void norm_avg_dev(const Tensor<T> &sum_ssq, const Tensor<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev_async(sum_ssq, avg_dev, nelems, eps);
    starpu_task_wait_for_all();
}

} // namespace nntile

