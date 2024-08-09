/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/fp16_to_fp32.hh
 * Convert fp16_t array into fp32_t array on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

void fp16_to_fp32_async(const Tensor<fp16_t> &src, const Tensor<fp32_t> &dst);

void fp16_to_fp32(const Tensor<fp16_t> &src, const Tensor<fp32_t> &dst);

} // namespace nntile::tensor
