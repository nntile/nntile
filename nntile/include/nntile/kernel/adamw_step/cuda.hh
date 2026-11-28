/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/adamw_step/cuda.hh
 * Fused AdamW step on CUDA buffers
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/kernel/cuda.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::adamw_step
{

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar beta_1,
        Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
        const T *grad, T *first_moment, T *second_moment, T *p)
    noexcept;

template<typename T>
void chunked(cudaStream_t stream, Index num_iter, Index num_elems,
        Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr,
        Scalar weight_decay,
        const T* const* grad_chunks, Index ngrad,
        T* const* first_moment_chunks, Index nfirst_moment,
        T* const* second_moment_chunks, Index nsecond_moment,
        T* const* p_chunks, Index np,
        Index elems_grad, Index elems_first_moment,
        Index elems_second_moment, Index elems_p,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::adamw_step
