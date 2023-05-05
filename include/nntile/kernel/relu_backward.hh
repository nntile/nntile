/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/relu_backward.hh
 * Backward ReLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-05
 * */

#pragma once

#include <nntile/kernel/relu_backward/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/relu_backward/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::relu
/*! Low-level implementations of backward ReLU operation
 * */
namespace relu
{

} // namespace relu
} // namespace kernel
} // namespace nntile

