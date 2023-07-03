/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/prod_fiber3/cuda.hh
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
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
namespace prod_fiber3
{

// Per-element product of a tensor and a broadcasted fiber on CPU
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, T alpha,
        const T *src1, const T *src2, T *dst)
    noexcept;

} // namespace prod_fiber3
} // namespace kernel
} // namespace nntile

