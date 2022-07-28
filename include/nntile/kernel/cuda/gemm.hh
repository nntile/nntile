/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cuda/gemm.hh
 * GEMM operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/constants.hh>

namespace nntile
{

template<typename T>
void gemm_kernel_cublas(TransOp transA, TransOp transB, Index m, Index n,
        Index k, T alpha, const T *A, const T *B, T beta, T *C);

template<typename T>
void gemm_starpu_cuda(void *buffers[], void *cl_args);

} // namespace nntile

