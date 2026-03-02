/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor_graph.hh
 * TensorGraph - graph operating on tensors, derives from BaseGraph.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/base_graph.hh>

namespace nntile::graph
{

//! Tensor graph - defines computation at tensor level
class TensorGraph : public BaseGraph<TensorGraph>
{
public:
    using DataNode = BaseDataNode<TensorGraph>;
    using OpNode = BaseOpNode<TensorGraph>;

    explicit TensorGraph(const std::string& name = "")
        : BaseGraph<TensorGraph>(name)
    {
    }

    //! Called by BaseGraph::add_op; Graph is friend of BaseOpNode so can set id_
    void assign_op_id(OpNode* op, NodeId id)
    {
        op->id_ = id;
    }

protected:
    const char* graph_type_name() const override
    {
        return "TensorGraph";
    }
};

} // namespace nntile::graph
