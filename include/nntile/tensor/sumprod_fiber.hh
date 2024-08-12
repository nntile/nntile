/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/sumprod_fiber.hh
 * Sums over fibers into a slice of a product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
void sumprod_fiber_async(Scalar alpha, const Tensor<T> &src1,
        const Tensor<T> &src2, Scalar beta, const Tensor<T> &dst, Index axis,
        int redux=0);

template<typename T>
void sumprod_fiber(Scalar alpha, const Tensor<T> &src1, const Tensor<T> &src2,
        Scalar beta, const Tensor<T> &dst, Index axis, int redux=0);

} // namespace nntile::tensor
