/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/scal/cuda.hh
 * Scal operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace scal
{

// Apply scal for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, const T* src, T* dst)
    noexcept;

} // namespace scal
} // namespace kernel
} // namespace nntile

