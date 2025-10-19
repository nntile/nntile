/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/multiply_slice_inplace/cpu.hh
 * CPU kernel for in-place multiplication of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::multiply_slice_inplace
{

//! In-place multiplication of a tensor and a broadcasted slice on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = beta * dst[i,l,j] * alpha * src[i,j]
 *
 * @param[in] m: Size of the first mode of dst
 * @param[in] n: Size of the second mode of dst
 * @param[in] k: Size of the third mode of dst
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input slice
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Resulting tensor
 * */
template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha, const T *src, Scalar beta,
        T *dst) noexcept;

} // namespace nntile::kernel::multiply_slice_inplace
