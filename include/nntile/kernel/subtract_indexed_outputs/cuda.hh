/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs/cuda.hh
 * subtract_indexed_outputs operation for buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
void cuda(cudaStream_t stream, Index n_labels, Index n_outputs, Scalar val,
        const int64_t *labels, T *dst)
    noexcept;

} // namespace nntile::kernel::subtract_indexed_outputs
