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
 * Individual ops in nn/ (TensorGraph backend),
 * this file includes them all.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/nn/add.hh>
#include <nntile/graph/nn/add_fiber.hh>
#include <nntile/graph/nn/add_slice.hh>
#include <nntile/graph/nn/clear.hh>
#include <nntile/graph/nn/concat.hh>
#include <nntile/graph/nn/cross_entropy.hh>
#include <nntile/graph/nn/embedding.hh>
#include <nntile/graph/nn/fill.hh>
#include <nntile/graph/nn/gemm.hh>
#include <nntile/graph/nn/gelu.hh>
#include <nntile/graph/nn/gelutanh.hh>
#include <nntile/graph/nn/mse_loss.hh>
#include <nntile/graph/nn/multiply.hh>
#include <nntile/graph/nn/multiply_fiber.hh>
#include <nntile/graph/nn/multiply_slice.hh>
#include <nntile/graph/nn/norm.hh>
#include <nntile/graph/nn/norm_fiber.hh>
#include <nntile/graph/nn/norm_slice.hh>
#include <nntile/graph/nn/relu.hh>
#include <nntile/graph/nn/rms_norm.hh>
#include <nntile/graph/nn/layer_norm.hh>
#include <nntile/graph/nn/rope.hh>
#include <nntile/graph/nn/sdpa_eager.hh>
#include <nntile/graph/nn/scale.hh>
#include <nntile/graph/nn/scale_fiber.hh>
#include <nntile/graph/nn/scale_slice.hh>
#include <nntile/graph/nn/silu.hh>
#include <nntile/graph/nn/softmax.hh>
#include <nntile/graph/nn/sgd_step.hh>
#include <nntile/graph/nn/adam_step.hh>
#include <nntile/graph/nn/adamw_step.hh>
#include <nntile/graph/nn/sum_fiber.hh>
#include <nntile/graph/nn/sum_slice.hh>
#include <nntile/graph/nn/transpose.hh>
