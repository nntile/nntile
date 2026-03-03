/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/conv2d_bwd_weight_inplace.hh
 * TensorGraph conv2d_bwd_weight_inplace: dC = alpha*conv_bwd(X,dY) + beta*dC
 *
 * @version 1.1.0
 * */

#pragma once

#include <array>
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Conv2D backward weight: dC = alpha*conv_bwd(X,dY) + beta*dC
struct TensorConv2dBwdWeightInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    TensorGraph::TensorNode* X = nullptr;
    TensorGraph::TensorNode* dY = nullptr;
    Scalar beta;
    TensorGraph::TensorNode* dC = nullptr;
    std::array<Index, 2> padding = {0, 0};
    std::array<Index, 2> stride = {1, 1};
    std::array<Index, 2> dilation = {1, 1};

    TensorConv2dBwdWeightInplaceOp() = default;
    TensorConv2dBwdWeightInplaceOp(Scalar alpha_,
                                   TensorGraph::TensorNode* X_,
                                   TensorGraph::TensorNode* dY_,
                                   Scalar beta_,
                                   TensorGraph::TensorNode* dC_,
                                   std::array<Index, 2> padding_,
                                   std::array<Index, 2> stride_,
                                   std::array<Index, 2> dilation_)
        : alpha(alpha_), X(X_), dY(dY_), beta(beta_), dC(dC_),
          padding(padding_), stride(stride_), dilation(dilation_)
    {
        inputs_ = {X, dY, dC};
        outputs_ = {dC};
    }

    std::string op_name() const override
    {
        return "CONV2D_BWD_WEIGHT_INPLACE";
    }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorConv2dBwdWeightInplaceOp>(*this);
    }
};

//! Conv2D backward weight: dC = alpha*conv_bwd(X,dY) + beta*dC
void conv2d_bwd_weight_inplace(Scalar alpha,
                               TensorGraph::TensorNode* X,
                               TensorGraph::TensorNode* dY,
                               Scalar beta,
                               TensorGraph::TensorNode* dC,
                               std::array<Index, 2> padding,
                               std::array<Index, 2> stride,
                               std::array<Index, 2> dilation);

} // namespace nntile::graph
