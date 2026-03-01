/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/conv2d_bwd_input_inplace.cc
 * Logical graph Conv2D backward input in-place operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/conv2d_bwd_input_inplace.hh"

// Include standard headers
#include <stdexcept>
#include <utility>
#include <array>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! 2D Convolution backward w.r.t. input: dX = alpha * conv2d_bwd_input(dY, C) + beta * dX
void conv2d_bwd_input_inplace(
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& c,
    LogicalGraph::TensorNode& dx,
    Scalar alpha,
    Scalar beta,
    std::array<Index, 2> padding,
    std::array<Index, 2> stride,
    std::array<Index, 2> dilation)
{
    if(&dy.graph() != &c.graph() || &dy.graph() != &dx.graph())
    {
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must belong to the same graph");
    }

    if(dy.dtype() != c.dtype() || dy.dtype() != dx.dtype())
    {
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: all tensors must have the same dtype");
    }

    auto attrs = std::make_shared<Conv2dAttrs>(Conv2dAttrs{alpha, beta, padding, stride, dilation});
    dy.graph().add_op(
        OpType::CONV2D_BWD_INPUT_INPLACE,
        attrs,
        {&dy, &c, &dx},
        {&dx}
    );
}

} // namespace nntile::graph
