/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_scalar.hh
 * Add scalar to elements from buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2022-12-01
 * */

#pragma once

#include <nntile/kernel/add_scalar/cpu.hh>
//#include <nntile/defs.h>
//#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/hypot/cuda.hh>
//#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::add_scalar
/*! Low-level implementations of computing add scalar to buffer elements
 * */
namespace add_scalar
{

} // namespace add_scalar
} // namespace kernel
} // namespace nntile

