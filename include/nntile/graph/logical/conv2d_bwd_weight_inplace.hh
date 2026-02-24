/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/conv2d_bwd_weight_inplace.hh
 * Logical graph Conv2D backward weight in-place operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>
#include <array>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! 2D Convolution backward w.r.t. weights: dC = alpha * conv2d_bwd_weight(X, dY) + beta * dC
//! @param x Input tensor (WHCN format)
//! @param dy Gradient of output tensor (WHCN format)
//! @param dc Gradient tensor to accumulate into (WHCN format)
//! @param alpha Scaling factor for backward result (default: 1.0)
//! @param beta Scaling factor for existing dc (default: 1.0)
//! @param padding Padding for height and width [pad_h, pad_w]
//! @param stride Stride for height and width [stride_h, stride_w]
//! @param dilation Dilation for height and width [dilation_h, dilation_w]
void conv2d_bwd_weight_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dc,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    std::array<Index, 2> padding = {0, 0},
    std::array<Index, 2> stride = {1, 1},
    std::array<Index, 2> dilation = {1, 1}
);

} // namespace nntile::graph
