/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/conv2d_bwd_input_inplace.hh
 * TensorGraph conv2d_bwd_input_inplace: dX = alpha*conv_bwd(dY,kernel) + beta*dX
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <array>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Conv2D backward input: dX = alpha*conv_bwd(dY,kernel) + beta*dX
struct TensorConv2dBwdInputInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    TensorGraph::TensorNode* dY = nullptr;
    TensorGraph::TensorNode* kernel = nullptr;
    Scalar beta;
    TensorGraph::TensorNode* dX = nullptr;
    std::array<Index, 2> padding = {0, 0};
    std::array<Index, 2> stride = {1, 1};
    std::array<Index, 2> dilation = {1, 1};

    TensorConv2dBwdInputInplaceOp() = default;
    TensorConv2dBwdInputInplaceOp(Scalar alpha_,
                                 TensorGraph::TensorNode* dY_,
                                 TensorGraph::TensorNode* kernel_,
                                 Scalar beta_,
                                 TensorGraph::TensorNode* dX_,
                                 std::array<Index, 2> padding_,
                                 std::array<Index, 2> stride_,
                                 std::array<Index, 2> dilation_)
        : alpha(alpha_), dY(dY_), kernel(kernel_), beta(beta_), dX(dX_),
          padding(padding_), stride(stride_), dilation(dilation_)
    {
        inputs_ = {dY, kernel, dX};
        outputs_ = {dX};
    }

    std::string op_name() const override
    {
        return "CONV2D_BWD_INPUT_INPLACE";
    }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorConv2dBwdInputInplaceOp>(*this);
    }
};

//! Conv2D backward input: dX = alpha*conv_bwd(dY,kernel) + beta*dX
void conv2d_bwd_input_inplace(Scalar alpha,
                              TensorGraph::TensorNode* dY,
                              TensorGraph::TensorNode* kernel,
                              Scalar beta,
                              TensorGraph::TensorNode* dX,
                              std::array<Index, 2> padding,
                              std::array<Index, 2> stride,
                              std::array<Index, 2> dilation);

} // namespace nntile::graph::tensor
