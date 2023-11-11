/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum/cuda.hh
 * total_sum_accum operation for buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-11-11
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile
{
namespace kernel
{
namespace total_sum_accum
{

template<typename T>
void cuda(cudaStream_t stream, T alpha, Index n_labels, Index n_outputs, const T* logsumexp, const T* src,
        const Index* labels, T *val)
    noexcept;

} // namespace total_sum_accum
} // namespace kernel
} // namespace nntile

