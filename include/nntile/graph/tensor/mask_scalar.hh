/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/mask_scalar.hh
 * TensorGraph mask_scalar operation: A[mask] = val
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Mask scalar operation: A[mask] = val
struct TensorMaskScalarOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* mask = nullptr;
    Scalar val;
    TensorGraph::TensorNode* A = nullptr;
    Index batch_ndim;

    TensorMaskScalarOp() = default;
    TensorMaskScalarOp(TensorGraph::TensorNode* mask_,
                      Scalar val_,
                      TensorGraph::TensorNode* A_,
                      Index batch_ndim_)
        : mask(mask_), val(val_), A(A_), batch_ndim(batch_ndim_)
    {
        inputs_ = {mask, A};
        outputs_ = {A};
    }

    std::string op_name() const override { return "MASK_SCALAR"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMaskScalarOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Mask scalar: A[mask] = val
void mask_scalar(TensorGraph::TensorNode* mask,
                 Scalar val,
                 TensorGraph::TensorNode* A,
                 Index batch_ndim);

} // namespace nntile::graph::tensor
