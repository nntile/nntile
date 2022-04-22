/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/randn.hh
 * Randn operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Asynchronous tensor-wise random generation operation
//
// @param[out] dst: Destination tensor
// @param[in] offset: Offset of the destination tensor in the underlying tensor
// @param[in] shape: Shape of the underlying tensor
// @param[in] stride: Stride of the underlying tensor
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
//
// Randomly fill the output tensor as if it is a part of the provided
// underlying tensor. The destination tensor shall be fully inside the
// underlying tensor.
template<typename T>
void randn_async(const Tensor<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1);

extern template
void randn_async(const Tensor<fp32_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp32_t mean=0, fp32_t stddev=1);

extern template
void randn_async(const Tensor<fp64_t> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, fp64_t mean=0, fp64_t stddev=1);

//! Asynchronous tensor-wise random generation operation
//
// @param[out] dst: Destination tensor
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
template<typename T>
void randn_async(const Tensor<T> &dst, unsigned long long seed, T mean=0,
        T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
}

//! Blocking version of tensor-wise random generation operation
//
// @param[out] dst: Destination tensor
// @param[in] offset: Offset of the destination tensor in the underlying tensor
// @param[in] shape: Shape of the underlying tensor
// @param[in] stride: Stride of the underlying tensor
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
//
// Randomly fill the output tensor as if it is a part of the provided
// underlying tensor. The destination tensor shall be fully inside the
// underlying tensor.
template<typename T>
void randn(const Tensor<T> &dst, const std::vector<Index> &offset,
        const std::vector<Index> &shape, const std::vector<Index> &stride,
        unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, offset, shape, stride, seed, mean, stddev);
    starpu_task_wait_for_all();
}

//! Blocking version of tensor-wise random generation operation
//
// @param[out] dst: Destination tensor
// @param[in] seed: Seed for the normal random distribution
// @param[in] mean: Average of the normal random distribution
// @param[in] stddev: Deviation of the normal random distribution
template<typename T>
void randn(const Tensor<T> &dst, unsigned long long seed, T mean=0, T stddev=1)
{
    randn_async<T>(dst, std::vector<Index>(dst.ndim), dst.shape, dst.stride,
            seed, mean, stddev);
    starpu_task_wait_for_all();
}

} // namespace nntile

