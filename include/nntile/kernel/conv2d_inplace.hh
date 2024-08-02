/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/conv2d_inplace.hh
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/defs.h>
#include <nntile/kernel/conv2d_inplace/cpu.hh>

//! @namespace nntile::kernel::conv2d_inplace
/*! Low-level implementations of conv2d_inplace operation
 * */
namespace nntile::kernel::conv2d_inplace
{

} // namespace nntile::kernel::conv2d_inplace