/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelu_backward/cuda.hh
 * Backward GeLU operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace gelu_backward
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x, const T *dy, T *dx)
    noexcept;

} // namespace gelu_backward
} // namespace kernel
} // namespace nntile

