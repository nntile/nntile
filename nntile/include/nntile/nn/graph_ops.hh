/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn/graph_ops.hh
 * NNGraph operations - free function overloads for NNGraph::TensorNode.
 *
 * Individual ops in nn/ops/ (TensorGraph backend);
 * this file includes them all.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/nn/ops/add.hh>
#include <nntile/nn/ops/add_fiber.hh>
#include <nntile/nn/ops/add_slice.hh>
#include <nntile/nn/ops/clear.hh>
#include <nntile/nn/ops/concat.hh>
#include <nntile/nn/ops/cross_entropy.hh>
#include <nntile/nn/ops/embedding.hh>
#include <nntile/nn/ops/fill.hh>
#include <nntile/nn/ops/gemm.hh>
#include <nntile/nn/ops/gelu.hh>
#include <nntile/nn/ops/gelutanh.hh>
#include <nntile/nn/ops/layer_norm.hh>
#include <nntile/nn/ops/mse_loss.hh>
#include <nntile/nn/ops/multiply.hh>
#include <nntile/nn/ops/multiply_fiber.hh>
#include <nntile/nn/ops/multiply_slice.hh>
#include <nntile/nn/ops/norm.hh>
#include <nntile/nn/ops/norm_fiber.hh>
#include <nntile/nn/ops/norm_slice.hh>
#include <nntile/nn/ops/relu.hh>
#include <nntile/nn/ops/rms_norm.hh>
#include <nntile/nn/ops/rope.hh>
#include <nntile/nn/ops/sdpa_eager.hh>
#include <nntile/nn/ops/sdpa_causal_mask.hh>
#include <nntile/nn/ops/scale.hh>
#include <nntile/nn/ops/scale_fiber.hh>
#include <nntile/nn/ops/scale_slice.hh>
#include <nntile/nn/ops/silu.hh>
#include <nntile/nn/ops/softmax.hh>
#include <nntile/nn/ops/sgd_step.hh>
#include <nntile/nn/ops/adam_step.hh>
#include <nntile/nn/ops/adamw_step.hh>
#include <nntile/nn/ops/sum_fiber.hh>
#include <nntile/nn/ops/sum_slice.hh>
#include <nntile/nn/ops/transpose.hh>
