/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile.hh
 * Convenience header for the NNTile TensorGraph stack (libnntile).
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/context.hh>
#include <nntile/core.hh>
#include <nntile/tensor.hh>
#include <nntile/tensor/phase_codec.hh>
#include <nntile/tile.hh>
#include <nntile/runtime.hh>
#include <nntile/execution_driver.hh>
#include <nntile/remote_execution_driver.hh>
