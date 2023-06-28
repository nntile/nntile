/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs.hh
 * Subtract a value from certain elements of a matrix
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#pragma once

#include <nntile/kernel/subtract_indexed_outputs/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
// #include <nntile/kernel/subtract_indexed_outputs/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::subtract_indexed_outputs
/*! Low-level implementations of subtraction given value from certain matrix
 * elements
 * */
namespace subtract_indexed_outputs
{

} // namespace subtract_indexed_outputs
} // namespace kernel
} // namespace nntile

