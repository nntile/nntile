/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                 (AIRI), Russia. All rights reserved.
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
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/op_arena.hh>
#include <nntile/tile/graph_data_node.hh>

namespace nntile
{

class Runtime;

//! Base class for TileGraph operations.
class TileGraph::OpNode
{
public:
    using NodeId = uint64_t;

    //! Max edges a shipped tile op uses inline (largest dynamic builder:
    //! torch batch-norm backward, 8). Beyond this the span spills.
    static constexpr std::size_t kMaxInlineInputs = 12;
    static constexpr std::size_t kMaxInlineOutputs = 4;

    using EdgeSpanIn = OpEdgeSpan<TileGraph::TileNode, kMaxInlineInputs>;
    using EdgeSpanOut = OpEdgeSpan<TileGraph::TileNode, kMaxInlineOutputs>;

    virtual ~OpNode() = default;

    virtual std::string op_name() const = 0;
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    NodeId id() const { return id_; }

    const EdgeSpanIn& inputs() const
    {
        return inputs_;
    }
    const EdgeSpanOut& outputs() const
    {
        return outputs_;
    }

    //! Cheap kind probes replacing per-op virtual ``op_name()`` string
    //! construction in hot compile passes (snapshot build, execution-order
    //! append, DCE seeding, invalidate scan). Default: not this kind.
    virtual bool is_tile_unregister() const { return false; }
    virtual bool is_torch_dispatch() const { return false; }

    virtual void execute(Runtime &runtime) const = 0;
    virtual std::shared_ptr<TileGraph::OpNode> clone() const = 0;

protected:
    OpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    EdgeSpanIn inputs_;
    EdgeSpanOut outputs_;

    friend class TileGraph;
};

} // namespace nntile
