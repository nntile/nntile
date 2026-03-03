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

// Include NNGraph operation overloads (one file per op)
#include <nntile/graph/nn/add.hh>
#include <nntile/graph/nn/add_fiber.hh>
#include <nntile/graph/nn/clear.hh>
#include <nntile/graph/nn/fill.hh>
#include <nntile/graph/nn/gemm.hh>
#include <nntile/graph/nn/gelu.hh>
#include <nntile/graph/nn/gelutanh.hh>
#include <nntile/graph/nn/relu.hh>
#include <nntile/graph/nn/silu.hh>
#include <nntile/graph/nn/softmax.hh>
#include <nntile/graph/nn/sum_fiber.hh>
