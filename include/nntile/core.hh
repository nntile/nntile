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

#include <nntile/core/defs.h>
#include <nntile/core/base_types.hh>
#include <nntile/core/constants.hh>
#include <nntile/core/context.hh>
#include <nntile/core/starpu.hh>
#ifndef STARPU_SIMGRID
#include <nntile/core/kernel.hh>
#endif // STARPU_SIMGRID
#include <nntile/core/tile.hh>
#include <nntile/core/tensor.hh>
#include <nntile/core/logger.hh>
