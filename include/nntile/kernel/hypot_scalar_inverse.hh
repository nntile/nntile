/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/hypot_scalar_inverse.hh
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-28
 * */

#pragma once

#include <nntile/kernel/hypot_scalar_inverse/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/hypot_scalar_inverse/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::hypot_scalar_inverse
/*! Low-level implementations of hypot_scalar_inverse operation
 * */
namespace hypot_scalar_inverse
{

} // namespace hypot_scalar_inverse
} // namespace kernel
} // namespace nntile

