/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nn_graph/graph_ops.hh
 * NNGraph operations - free function overloads for NNGraph::TensorNode.
 *
 * Individual ops in nn/ops/ (TensorGraph backend);
 * this file includes them all.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/nn_graph/ops/add.hh>
#include <nntile/nn_graph/ops/add_fiber.hh>
#include <nntile/nn_graph/ops/add_slice.hh>
#include <nntile/nn_graph/ops/clear.hh>
#include <nntile/nn_graph/ops/concat.hh>
#include <nntile/nn_graph/ops/cross_entropy.hh>
#include <nntile/nn_graph/ops/embedding.hh>
#include <nntile/nn_graph/ops/fill.hh>
#include <nntile/nn_graph/ops/gemm.hh>
#include <nntile/nn_graph/ops/gelu.hh>
#include <nntile/nn_graph/ops/gelutanh.hh>
#include <nntile/nn_graph/ops/layer_norm.hh>
#include <nntile/nn_graph/ops/mse_loss.hh>
#include <nntile/nn_graph/ops/multiply.hh>
#include <nntile/nn_graph/ops/multiply_fiber.hh>
#include <nntile/nn_graph/ops/multiply_slice.hh>
#include <nntile/nn_graph/ops/norm.hh>
#include <nntile/nn_graph/ops/norm_fiber.hh>
#include <nntile/nn_graph/ops/norm_slice.hh>
#include <nntile/nn_graph/ops/relu.hh>
#include <nntile/nn_graph/ops/rms_norm.hh>
#include <nntile/nn_graph/ops/rope.hh>
#include <nntile/nn_graph/ops/sdpa_eager.hh>
#include <nntile/nn_graph/ops/sdpa_causal_mask.hh>
#include <nntile/nn_graph/ops/scale.hh>
#include <nntile/nn_graph/ops/scale_fiber.hh>
#include <nntile/nn_graph/ops/scale_slice.hh>
#include <nntile/nn_graph/ops/silu.hh>
#include <nntile/nn_graph/ops/softmax.hh>
#include <nntile/nn_graph/ops/sgd_step.hh>
#include <nntile/nn_graph/ops/adam_step.hh>
#include <nntile/nn_graph/ops/adamw_step.hh>
#include <nntile/nn_graph/ops/sum_fiber.hh>
#include <nntile/nn_graph/ops/sum_slice.hh>
#include <nntile/nn_graph/ops/transpose.hh>
