/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding_backward/cuda.hh
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::embedding_backward
{

//! vocab = beta*vocab + alpha*scatter(embed); beta in {0,1}
/*! If beta=0, the entire vocab buffer (vocab_nelems) is zeroed first, then
 * indexed rows accumulate alpha*embed. If beta=1, only accumulate.
 */
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index vocab_nelems, Scalar alpha, Scalar beta,
        const int64_t *index, const T *embed, T *vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
