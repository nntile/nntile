/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module.hh
 * Convenience header for NNTile module system.
 *
 * @version 1.1.0
 * */

#pragma once

// Include related NNTile headers
#include <nntile/module/module.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/gelu.hh>
#include <nntile/module/mlp.hh>
#include <nntile/module/mse_loss.hh>
#include <nntile/module/sdpa.hh>
