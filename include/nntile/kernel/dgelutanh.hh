/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/dgelutanh.hh
 * Derivative of approximated GeLU low-level kernels
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/kernel/dgelutanh/cpu.hh>
#include <nntile/defs.h>
#ifdef NNTILE_USE_CUDA
#include <nntile/kernel/dgelutanh/cuda.hh>
#endif // NNTILE_USE_CUDA

//! @namespace nntile::kernel::dgelu
/*! Low-level implementations of derivative of approximate GeLU operation
 * */
namespace nntile::kernel::dgelutanh
{

} // namespace nntile::kernel::dgelutanh
