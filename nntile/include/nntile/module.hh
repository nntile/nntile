/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module.hh
 * Convenience header for NNTile module classes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/module/module.hh>
#include <nntile/module/activation.hh>
#include <nntile/module/linear.hh>
#include <nntile/module/embedding.hh>
#include <nntile/module/gated_mlp.hh>
#include <nntile/module/mlp.hh>
#include <nntile/module/rms_norm.hh>
#include <nntile/module/layer_norm.hh>
