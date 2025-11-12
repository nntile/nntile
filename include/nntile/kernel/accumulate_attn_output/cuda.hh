/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/accumulate_attn_output/cuda.hh
 * Accumulate attention outputs on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::accumulate_attn_output
{

// Accumulate attention outputs on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index seq, Index batch, const fp32_t *src_lse,
        const T *src_attn, fp32_t *dst_lse, T *dst_attn)
    noexcept;

} // namespace nntile::kernel::accumulate_attn_output
