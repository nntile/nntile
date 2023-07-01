/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh_inplace/cuda.hh
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace gelutanh_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept;

} // namespace gelutanh_inplace
} // namespace kernel
} // namespace nntile

