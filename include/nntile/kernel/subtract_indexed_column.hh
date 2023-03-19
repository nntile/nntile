/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_column.hh
 * Low-level kernels to subtract value from indexed column
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#pragma once

#include <nntile/kernel/subtract_indexed_column/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
// #include <nntile/kernel/subtract_indexed_column/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::subtract_indexed_column
/*! Low-level implementations of subtraction given value from the indexed matrix column 
 * */
namespace subtract_indexed_column
{

} // namespace subtract_indexed_column
} // namespace kernel
} // namespace nntile

