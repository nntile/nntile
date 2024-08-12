/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/adam_step/cuda.hh
 * Fused Adam step on CUDA buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::adam_step
{

template<typename T>
void cuda(cudaStream_t stream, Index num_iter, Index num_elems, Scalar beta_1,
        Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay,
        const T *grad, T *first_moment, T *second_moment, T *p)
    noexcept;

} // namespace nntile::kernel::adam_step
