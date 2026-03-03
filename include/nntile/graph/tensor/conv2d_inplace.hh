/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/conv2d_inplace.hh
 * TensorGraph conv2d_inplace: Y = alpha*conv(X,C) + beta*Y
 *
 * @version 1.1.0
 * */

#pragma once

#include <array>
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Conv2D forward: Y = alpha*conv(X,C) + beta*Y
struct TensorConv2dInplaceOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    TensorGraph::TensorNode* X = nullptr;
    TensorGraph::TensorNode* C = nullptr;
    Scalar beta = 0.0;
    TensorGraph::TensorNode* Y = nullptr;
    std::array<Index, 2> padding = {0, 0};
    std::array<Index, 2> stride = {1, 1};
    std::array<Index, 2> dilation = {1, 1};

    TensorConv2dInplaceOp() = default;
    TensorConv2dInplaceOp(Scalar alpha_, TensorGraph::TensorNode* X_,
                          TensorGraph::TensorNode* C_, Scalar beta_,
                          TensorGraph::TensorNode* Y_,
                          std::array<Index, 2> padding_,
                          std::array<Index, 2> stride_,
                          std::array<Index, 2> dilation_)
        : alpha(alpha_), X(X_), C(C_), beta(beta_), Y(Y_),
          padding(padding_), stride(stride_), dilation(dilation_)
    {
        inputs_ = {X, C, Y};
        outputs_ = {Y};
    }

    std::string op_name() const override { return "CONV2D_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorConv2dInplaceOp>(*this);
    }
};

//! Conv2D forward: Y = alpha*conv(X,C) + beta*Y
void conv2d_inplace(Scalar alpha,
                    TensorGraph::TensorNode* X,
                    TensorGraph::TensorNode* C,
                    Scalar beta,
                    TensorGraph::TensorNode* Y,
                    std::array<Index, 2> padding,
                    std::array<Index, 2> stride,
                    std::array<Index, 2> dilation);

} // namespace nntile::graph
