/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_softmax_gemm_backward_dq_dk.hh
 * Header for computing gradients dQ and dK of softmax(A)V
 *
 * @version 1.1.0
 * */
#pragma once

#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/flash_softmax_gemm_backward_dq_dk/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
/*! Low-level implementations to compute gradients dQ and dK of softmax(A)V
 * */
namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
{

} // namespace nntile::kernel::flash_softmax_gemm_backward_dq_dk
