/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/flash_sdpa_bwd_cudnn.hh
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/flash_sdpa_bwd_cudnn/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::flash_sdpa_bwd_cudnn
/*! Low-level implementations of flash attention SDPA backward pass using cuDNN
 * */
namespace nntile::kernel::flash_sdpa_bwd_cudnn
{

} // namespace nntile::kernel::flash_sdpa_bwd_cudnn
