/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/total_sum_accum.hh
 * TensorGraph total_sum_accum operation
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Total sum accumulating: (alpha, logsumexp, src, class_labels, val, ignore_index)
struct TensorTotalSumAccumOp : TensorGraph::OpNode
{
    Scalar alpha;
    Index ignore_index;
    TensorGraph::TensorNode* logsumexp = nullptr;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* class_labels = nullptr;
    TensorGraph::TensorNode* val = nullptr;

    TensorTotalSumAccumOp() = default;
    TensorTotalSumAccumOp(
        Scalar alpha_,
        TensorGraph::TensorNode* logsumexp_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* class_labels_,
        TensorGraph::TensorNode* val_,
        Index ignore_index_)
        : alpha(alpha_), ignore_index(ignore_index_)
        , logsumexp(logsumexp_), src(src_)
        , class_labels(class_labels_), val(val_)
    {
        inputs_ = {logsumexp, src, class_labels, val};
        outputs_ = {val};
    }

    std::string op_name() const override { return "TOTAL_SUM_ACCUM"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTotalSumAccumOp>(*this);
    }
};

void total_sum_accum(
    Scalar alpha,
    TensorGraph::TensorNode* logsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* class_labels,
    TensorGraph::TensorNode* val,
    Index ignore_index = -1);

} // namespace nntile::graph::tensor
