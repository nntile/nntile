/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs/cuda.hh
 * subtract_indexed_outputs operation for buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_outputs
{

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, T val, const Index* labels, T *dst)
    noexcept;

} // namespace subtract_indexed_outputs
} // namespace kernel
} // namespace nntile

