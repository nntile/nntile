/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/fp32_to_fp16.hh
 * Convert fp32_t array into fp16_t array on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-04
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{
namespace tensor
{

void fp32_to_fp16_async(const Tensor<fp32_t> &src, const Tensor<fp16_t> &dst);

void fp32_to_fp16(const Tensor<fp32_t> &src, const Tensor<fp16_t> &dst);

} // namespace tensor
} // namespace nntile

