/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/drelu/cuda.hh
 * Derivative of ReLU operation on a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace drelu
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept;

} // namespace drelu
} // namespace kernel
} // namespace nntile

