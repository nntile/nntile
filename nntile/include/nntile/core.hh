/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/core.hh
 * Core API umbrella header (kernels, StarPU, tile, tensor).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>
#include <nntile/base_types.hh>
#include <nntile/constants.hh>
#include <nntile/context.hh>
#include <nntile/starpu.hh>
#ifndef STARPU_SIMGRID
#include <nntile/kernel.hh>
#endif // STARPU_SIMGRID
#include <nntile/tile.hh>
#include <nntile/tensor.hh>
#include <nntile/logger.hh>
