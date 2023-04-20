/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/biasprod_outer.hh
 * Bias-like product along outer axes low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-20
 * */

#pragma once

#include <nntile/kernel/biasprod_outer/cpu.hh>
//#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
//#include <nntile/kernel/biasprod_outer/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::biasprod_outer
/*! Low-level implementations of biasprod_outer operation
 * */
namespace biasprod_outer
{

} // namespace biasprod_outer
} // namespace kernel
} // namespace nntile

