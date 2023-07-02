/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/softmax_inplace.hh
 * Low-level kernels to softmax_inplace along axis
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-02
 * */

#pragma once

#include <nntile/kernel/softmax_inplace/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/softmax_inplace/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::softmax_inplace
/*! Low-level implementations of softmax_inplace operation
 * */
namespace softmax_inplace
{

} // namespace softmax_inplace
} // namespace kernel
} // namespace nntile

