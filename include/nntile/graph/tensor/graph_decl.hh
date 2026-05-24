/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_decl.hh
 * Tensor graph: symbolic computation built from simple tensor-wise operations
 * (data nodes + ops).  Autograd and ``backward()`` belong to ``NNGraph``,
 * which may record further tensor ops into this graph.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

struct AxisDescriptor;

//! Tensor graph - defines computation at tensor level (simple tensor ops).
class TensorGraph
{
  public:
    class TensorNode;
    class OpNode;
    using NodeId = uint64_t;

    explicit TensorGraph(const std::string &name = "") : name_(name) {}

    //! Create a data node with fresh axis descriptors and empty label.
    //! Call ``TensorNode::set_name`` for a debugging label.
    TensorNode *data(
        std::vector<Index> shape, DataType dtype = DataType::FP32);

    //! Add an operation to the graph
    void add_op(std::shared_ptr<TensorGraph::OpNode> op_node,
        const std::string &name = "");

    //! Collect unique axis groups across all tensors in the graph.
    //! Returns a vector of pointers to the distinct AxisDescriptors.
    std::vector<AxisDescriptor *> axis_groups() const;

    //! Number of axis groups that have no tiling set.
    size_t num_untiled_groups() const;

    // Queries
    const std::string &name() const { return name_; }
    size_t num_data() const { return data_.size(); }
    size_t num_ops() const { return ops_.size(); }

    //! Rename a data node (labels only; identity is the pointer).
    void rename_data_node(TensorNode *node, std::string new_name);

    std::vector<std::string> data_names() const;

    const std::vector<std::unique_ptr<TensorNode>> &tensor_nodes() const
    {
        return data_;
    }

    const std::vector<std::shared_ptr<TensorGraph::OpNode>> &ops() const
    {
        return ops_;
    }

    std::string to_string() const;
    std::string to_mermaid() const;

    //! Immutable view of one compile phase for incremental lowering.
    //! `op_begin`/`op_end` index into `ops()` at seal time; safe for overlap
    //! with execution because later `add_op` only appends.
    struct PhaseSnapshot
    {
        size_t op_begin = 0;
        size_t op_end = 0;
        //! Persistent tensors at seal time (input/output marks), unioned with
        //! op inputs/outputs when lowering (see ``collect_phase_tensors``).
        std::vector<TensorNode const *> carried_tensors;

        bool empty() const { return op_begin >= op_end; }
    };

    //! Seal ops [phase_seal_cursor_, num_ops()) into a snapshot and advance
    //! the cursor.  Carries every data node marked ``mark_input`` or
    //! ``mark_output`` (persistent across phases); phase temporaries are only
    //! those touched by ops in this phase and need no marks.
    PhaseSnapshot seal_phase();

    //! Same with an explicit carried list (overrides automatic marks).
    PhaseSnapshot seal_phase(std::vector<TensorNode const *> carried);

    void reset_phase_seal_cursor();

    size_t phase_seal_cursor() const { return phase_seal_cursor_; }

  private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> data_;
    std::vector<std::shared_ptr<TensorGraph::OpNode>> ops_;

    NodeId next_data_id_ = 0;
    NodeId next_op_id_ = 0;
    size_t phase_seal_cursor_ = 0;
};

//! One sealed slice of a ``TensorGraph``, ready for optional transforms and
//! lowering.  ``tensor_graph`` points at the owning graph (typically the inner
//! graph of an ``NNGraph``); ``snapshot`` indexes ``tensor_graph->ops()``.
struct FinishedTensorPhase
{
    TensorGraph const *tensor_graph = nullptr;
    TensorGraph::PhaseSnapshot snapshot;
};

//! Record of one lowered phase: tensor op slice and matching tile op span on a
//! shared ``TileGraph``.  Tensor node pointers in ``tensor_phase`` refer into
//! the live ``TensorGraph`` (append-only); valid while that graph outlives
//! this entry.
struct TensorPhaseArchiveEntry
{
    TensorGraph::PhaseSnapshot tensor_phase{};
    std::size_t tile_op_begin = 0;
    std::size_t tile_op_end = 0;
};

} // namespace nntile::graph
