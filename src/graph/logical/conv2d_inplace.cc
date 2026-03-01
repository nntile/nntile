/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/conv2d_inplace.cc
 * Logical graph Conv2D in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/conv2d_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <array>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! 2D Convolution forward: Y = alpha * conv2d(X, C) + beta * Y
void conv2d_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& c,
    LogicalGraph::TensorNode& y,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&x.graph() != &c.graph() || &x.graph() != &y.graph())
    {
        throw std::invalid_argument(
            "conv2d_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != c.dtype() || x.dtype() != y.dtype())
    {
        throw std::invalid_argument(
            "conv2d_inplace: all tensors must have the same dtype");
    }

    auto attrs = std::make_shared<Conv2dAttrs>(Conv2dAttrs{alpha, beta, padding, stride, dilation});
    x.graph().add_op(
        OpType::CONV2D_INPLACE,
        attrs,
        {&x, &c, &y},
        {&y}
    );
}

} // namespace nntile::graph
