/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/conv2d_inplace.hh
 * Logical graph Conv2D in-place operation.
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

//! 2D Convolution forward: Y = alpha * conv2d(X, C) + beta * Y
//! @param x Input tensor (WHCN format)
//! @param c Kernel tensor (WHCN format)
//! @param y Output tensor to accumulate into (WHCN format)
//! @param alpha Scaling factor for convolution result (default: 1.0)
//! @param beta Scaling factor for existing y (default: 1.0)
//! @param padding Padding for height and width [pad_h, pad_w]
//! @param stride Stride for height and width [stride_h, stride_w]
//! @param dilation Dilation for height and width [dilation_h, dilation_w]
void conv2d_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& c,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    std::array<Index, 2> padding = {0, 0},
    std::array<Index, 2> stride = {1, 1},
    std::array<Index, 2> dilation = {1, 1}
);

} // namespace nntile::graph
