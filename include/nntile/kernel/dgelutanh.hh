/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/dgelu.hh
 * Derivative of approximated GeLU low-level kernels
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-25
 * */

#pragma once

#include <nntile/kernel/dgelutanh/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/dgelutanh/cuda.hh>
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace kernel
{
//! @namespace nntile::kernel::dgelu
/*! Low-level implementations of derivative of approximate GeLU operation
 * */
namespace dgelutanh
{

} // namespace dgelutanh
} // namespace kernel
} // namespace nntile

