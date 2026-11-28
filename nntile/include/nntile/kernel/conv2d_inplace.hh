/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/conv2d_inplace.hh
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#pragma once

#include <nntile/kernel/conv2d_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/conv2d_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::conv2d_inplace
/*! Low-level implementations of conv2d_inplace operation
 * */
namespace nntile::kernel::conv2d_inplace
{

} // namespace nntile::kernel::conv2d_inplace
