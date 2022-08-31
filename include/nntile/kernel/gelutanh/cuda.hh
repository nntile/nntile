/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gelutanh/cuda.hh
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace gelutanh
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept;

} // namespace gelutanh
} // namespace kernel
} // namespace nntile

