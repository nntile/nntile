/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/subtract_indexed_outputs.hh
 * TensorGraph subtract_indexed_outputs: dst[labels[i]] -= val
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Subtract indexed outputs: dst[labels[i]] -= val
struct TensorSubtractIndexedOutputsOp : TensorGraph::OpNode
{
    Scalar val;
    TensorGraph::TensorNode* labels = nullptr;
    TensorGraph::TensorNode* dst = nullptr;
    Index ignore_index = -1;

    TensorSubtractIndexedOutputsOp() = default;
    TensorSubtractIndexedOutputsOp(Scalar val_,
                                  TensorGraph::TensorNode* labels_,
                                  TensorGraph::TensorNode* dst_,
                                  Index ignore_index_)
        : val(val_), labels(labels_), dst(dst_), ignore_index(ignore_index_)
    {
        inputs_ = {labels, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SUBTRACT_INDEXED_OUTPUTS"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSubtractIndexedOutputsOp>(*this);
    }
};

//! Subtract indexed outputs: dst[labels[i]] -= val
void subtract_indexed_outputs(Scalar val,
                             TensorGraph::TensorNode* labels,
                             TensorGraph::TensorNode* dst,
                             Index ignore_index);

} // namespace nntile::graph
