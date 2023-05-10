/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/pow/cuda.hh
 * Power operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace pow
{

// Power operation on a CUDA buffer
template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, T exp, T *data)
    noexcept;

} // namespace pow
} // namespace kernel
} // namespace nntile

