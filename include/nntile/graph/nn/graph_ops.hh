/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/graph_ops.hh
 * NNGraph operations - free function overloads for NNGraph::TensorNode.
 *
 * Individual ops in nn/ops/ (TensorGraph backend);
 * this file includes them all.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/nn/ops/add.hh>
#include <nntile/graph/nn/ops/add_fiber.hh>
#include <nntile/graph/nn/ops/add_slice.hh>
#include <nntile/graph/nn/ops/clear.hh>
#include <nntile/graph/nn/ops/concat.hh>
#include <nntile/graph/nn/ops/cross_entropy.hh>
#include <nntile/graph/nn/ops/embedding.hh>
#include <nntile/graph/nn/ops/fill.hh>
#include <nntile/graph/nn/ops/gemm.hh>
#include <nntile/graph/nn/ops/gelu.hh>
#include <nntile/graph/nn/ops/gelutanh.hh>
#include <nntile/graph/nn/ops/mse_loss.hh>
#include <nntile/graph/nn/ops/multiply.hh>
#include <nntile/graph/nn/ops/multiply_fiber.hh>
#include <nntile/graph/nn/ops/multiply_slice.hh>
#include <nntile/graph/nn/ops/norm.hh>
#include <nntile/graph/nn/ops/norm_fiber.hh>
#include <nntile/graph/nn/ops/norm_slice.hh>
#include <nntile/graph/nn/ops/relu.hh>
#include <nntile/graph/nn/ops/rms_norm.hh>
#include <nntile/graph/nn/ops/rope.hh>
#include <nntile/graph/nn/ops/sdpa_eager.hh>
#include <nntile/graph/nn/ops/scale.hh>
#include <nntile/graph/nn/ops/scale_fiber.hh>
#include <nntile/graph/nn/ops/scale_slice.hh>
#include <nntile/graph/nn/ops/silu.hh>
#include <nntile/graph/nn/ops/softmax.hh>
#include <nntile/graph/nn/ops/sgd_step.hh>
#include <nntile/graph/nn/ops/adam_step.hh>
#include <nntile/graph/nn/ops/adamw_step.hh>
#include <nntile/graph/nn/ops/sum_fiber.hh>
#include <nntile/graph/nn/ops/sum_slice.hh>
#include <nntile/graph/nn/ops/transpose.hh>
