/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/mask_scalar.hh
 * Low-level kernel to mask operation with given scalar
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#pragma once

#include <nntile/kernel/mask_scalar/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/mask_scalar/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::mask_scalar
/*! Low-level implementations of mask scalar operation
 * */
namespace mask_scalar
{

} // namespace mask_scalar
} // namespace kernel
} // namespace nntile

