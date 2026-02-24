/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/conv2d_bwd_weight_inplace.cc
 * Logical graph Conv2D backward weight in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/conv2d_bwd_weight_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <array>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! 2D Convolution backward w.r.t. weights: dC = alpha * conv2d_bwd_weight(X, dY) + beta * dC
void conv2d_bwd_weight_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dc,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&x.graph() != &dy.graph() || &x.graph() != &dc.graph())
    {
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: tensors must belong to the same graph");
    }

    if(x.dtype() != dy.dtype() || x.dtype() != dc.dtype())
    {
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: all tensors must have the same dtype");
    }

    OpAttrs attrs = Conv2dAttrs{alpha, beta, padding, stride, dilation};
    x.graph().add_op(
        OpType::CONV2D_BWD_WEIGHT_INPLACE,
        attrs,
        {&x, &dy, &dc},
        {&dc}
    );
}

} // namespace nntile::graph
