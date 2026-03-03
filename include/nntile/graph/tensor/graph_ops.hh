/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_ops.hh
 * TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/add.hh>
#include <nntile/graph/tensor/add_fiber.hh>
#include <nntile/graph/tensor/add_fiber_inplace.hh>
#include <nntile/graph/tensor/add_inplace.hh>
#include <nntile/graph/tensor/clear.hh>
#include <nntile/graph/tensor/fill.hh>
#include <nntile/graph/tensor/gemm.hh>
#include <nntile/graph/tensor/gelu.hh>
#include <nntile/graph/tensor/gelu_backward.hh>
#include <nntile/graph/tensor/multiply.hh>
#include <nntile/graph/tensor/norm.hh>
#include <nntile/graph/tensor/sum_fiber.hh>
