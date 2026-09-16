/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/graph_op_node.hh
 * TileGraph::OpNode - base class for TileGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <memory>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/tile/graph_data_node.hh>

namespace nntile
{

class Runtime;

//! Base class for TileGraph operations.
class TileGraph::OpNode
{
public:
    using NodeId = uint64_t;

    virtual ~OpNode() = default;

    virtual std::string op_name() const = 0;
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    NodeId id() const { return id_; }

    const std::vector<TileGraph::TileNode*>& inputs() const
    {
        return inputs_;
    }
    const std::vector<TileGraph::TileNode*>& outputs() const
    {
        return outputs_;
    }

    virtual void execute(Runtime &runtime) const = 0;
    virtual std::shared_ptr<TileGraph::OpNode> clone() const = 0;

    //! Logical worker 0..N-1, or -1 for StarPU dynamic placement.
    int device_hint() const { return device_hint_; }
    void set_device_hint(int hint) { device_hint_ = hint; }

    //! Rewire ``from`` to ``to`` in input/output lists and named aliases.
    void replace_tile(TileNode *from, TileNode *to);

    virtual void replace_named_tile(TileNode *from, TileNode *to)
    {
        (void)from;
        (void)to;
    }

protected:
    OpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<TileGraph::TileNode*> inputs_;
    std::vector<TileGraph::TileNode*> outputs_;
    int device_hint_ = -1;

    friend class TileGraph;
};

inline void TileGraph::OpNode::replace_tile(TileNode *from, TileNode *to)
{
    if (from == nullptr || from == to)
    {
        return;
    }
    for (TileNode *&p : inputs_)
    {
        if (p == from)
        {
            p = to;
        }
    }
    for (TileNode *&p : outputs_)
    {
        if (p == from)
        {
            p = to;
        }
    }
    replace_named_tile(from, to);
}

} // namespace nntile
