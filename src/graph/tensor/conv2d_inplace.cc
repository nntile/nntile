/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_inplace.cc
 * TensorGraph conv2d_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/conv2d_inplace.hh"

namespace nntile::graph::tensor
{



void conv2d_inplace(Scalar alpha,
                    TensorGraph::TensorNode* X,
                    TensorGraph::TensorNode* C,
                    Scalar beta,
                    TensorGraph::TensorNode* Y,
                    std::array<Index, 2> padding,
                    std::array<Index, 2> stride,
                    std::array<Index, 2> dilation)
{
    if(X == nullptr || C == nullptr || Y == nullptr)
        throw std::invalid_argument("conv2d_inplace: tensors must be non-null");
    if(X->graph() != C->graph() || C->graph() != Y->graph())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must belong to same graph");
    if(X->dtype() != C->dtype() || C->dtype() != Y->dtype())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dInplaceOp>(
        alpha, X, C, beta, Y, padding, stride, dilation);
    Y->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
