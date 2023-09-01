/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/hypot/cuda.hh
 * hypot operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace hypot
{

// Apply hypot for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, const T* src, T beta,
        T* dst)
    noexcept;

} // namespace hypot
} // namespace kernel
} // namespace nntile

