/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/fp32_to_fp16.hh
 * Convert fp32_t array into fp16_t array on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

void fp32_to_fp16_async(const Tensor<fp32_t> &src, const Tensor<fp16_t> &dst);

void fp32_to_fp16(const Tensor<fp32_t> &src, const Tensor<fp16_t> &dst);

} // namespace nntile::tensor
