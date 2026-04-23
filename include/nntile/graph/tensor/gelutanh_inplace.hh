/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gelutanh_inplace.hh
 * TensorGraph gelutanh_inplace operation: dst = gelutanh(dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! GeLUTanh in-place operation: dst = gelutanh(dst)
struct TensorGelutanhInplaceOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* dst = nullptr;

    TensorGelutanhInplaceOp() = default;
    TensorGelutanhInplaceOp(TensorGraph::TensorNode* dst_)
        : dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "GELUTANH_INPLACE"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGelutanhInplaceOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

void gelutanh_inplace(TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
