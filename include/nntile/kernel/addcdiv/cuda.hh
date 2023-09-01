/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/addcdiv/cuda.hh
 * Addcdiv operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-29
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace addcdiv
{

template<typename T>
void cuda(cudaStream_t stream, T val, T eps, Index nelems, const T *nom, const T* denom, T *res)
    noexcept;

} // namespace addcdiv
} // namespace kernel
} // namespace nntile

