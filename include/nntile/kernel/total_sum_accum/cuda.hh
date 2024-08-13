/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum/cuda.hh
 * total_sum_accum operation for buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::total_sum_accum
{

template<typename T>
void cuda(cudaStream_t stream, Scalar alpha, Index n_labels, Index n_outputs,
        const T *logsumexp, const T *src, const int64_t *labels, float *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
