/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_fiber/cuda.hh
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-20
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace add_fiber
{

// Per-element addition of a tensor and a broadcasted fiber on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src, T beta, T *dst)
    noexcept;

} // namespace add_fiber
} // namespace kernel
} // namespace nntile

