/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_scalar_scaled_inplace.hh
 * TensorGraph add_scalar_scaled_inplace: dst = alpha * scalar_val * src + beta * dst
 * where scalar_val is read from scalar_tensor (0-dimensional) at runtime.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Add scalar scaled in-place operation at tensor level
struct TensorAddScalarScaledInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* scalar_tensor = nullptr;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorAddScalarScaledInplaceOp() = default;
    TensorAddScalarScaledInplaceOp(
        TensorGraph::TensorNode* scalar_tensor_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_)
        , scalar_tensor(scalar_tensor_), src(src_), dst(dst_)
    {
        inputs_ = {scalar_tensor, src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "ADD_SCALAR_SCALED_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddScalarScaledInplaceOp>(*this);
    }
};

//! Add scalar scaled in-place: dst = alpha * scalar_tensor * src + beta * dst
void add_scalar_scaled_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* scalar_tensor,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
