/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile.hh
 * Main header file of the NNTile framework.
 *
 * @version 1.1.0
 * */

#pragma once

// Base data types (e.g., Index, fp64_t, fp32_t)
#include <nntile/base_types.hh>

// Constants (e.g., transposition for gemm)
#include <nntile/constants.hh>

// StarPU init/deinit and data handles
#include <nntile/starpu.hh>

// Kernel-level operations
#ifndef STARPU_SIMGRID
#include <nntile/kernel.hh>
#endif // STARPU_SIMGRID

// Fortran-contiguous tile with its operations
#include <nntile/tile.hh>

// Tensor as a set of tiles with its operations
#include <nntile/tensor.hh>

// Layers
//#include <nntile/layer.hh>

// Logger thread to log activities
#include <nntile/logger.hh>
