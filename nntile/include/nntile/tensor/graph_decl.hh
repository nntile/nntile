/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/graph_decl.hh
 * Tensor graph: symbolic computation built from simple tensor-wise operations
 * (data nodes + ops). Autograd and ``backward()`` belong to PyTorch /
 * libtorch_nntile, which record tensor ops into this graph.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/dtype.hh>

namespace nntile
{

struct AxisDescriptor;
class TensorRef;

//! Tensor graph - defines computation at tensor level (simple tensor ops).
class TensorGraph
{
  public:
    class TensorNode;
    class OpNode;
    using NodeId = uint64_t;

    explicit TensorGraph(const std::string &name = "") : name_(name) {}

    //! Create a data node with fresh axis descriptors and empty label.
    //! Does not create a ``TensorRef`` (for ephemeral staging / internal use).
    TensorNode *emplace_data(
        std::vector<Index> shape, DataType dtype = DataType::FP32);

    //! Create a data node and return a ``TensorRef``; keep it alive while the
    //! tensor is accessible. Call ``TensorNode::set_name`` for a debugging label.
    TensorRef data(
        std::vector<Index> shape, DataType dtype = DataType::FP32);

    //! Add an operation to the graph
    void add_op(std::shared_ptr<TensorGraph::OpNode> op_node,
        const std::string &name = "");

    //! Insert operations at the front of the op list (before existing ops).
    void prepend_ops(std::vector<std::shared_ptr<OpNode>> op_nodes);

    //! Live axis groups (maintained). Updated on ``emplace_data``,
    //! ``merge_axis``, and TensorNode GC — never rebuilt from ``data_``.
    std::vector<AxisDescriptor *> axis_groups() const;

    //! True if any live axis group has tiling set. O(|groups|), no copy.
    bool has_tiled_axis_group() const;

    //! Number of axis groups that have no tiling set.
    size_t num_untiled_groups() const;

    // Queries
    const std::string &name() const { return name_; }
    //! High-water slot count (includes GC holes). NodeIds are never reused.
    size_t num_data() const { return data_.size(); }
    //! Non-null data nodes remaining after GC.
    size_t num_live_data() const;
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
        //! Tensors needed for this phase: op inputs/outputs plus live
        //! ``mark_output`` nodes (see ``collect_phase_tensors``). Historical
        //! ``mark_input`` nodes that are unused this phase are omitted so
        //! incremental compile stays O(phase), not O(session).
        std::vector<TensorNode const *> carried_tensors;

        bool empty() const { return op_begin >= op_end; }
    };

    //! Seal ops [phase_seal_cursor_, num_ops()) into a snapshot and advance
    //! the cursor. Carries tensors referenced by those ops plus live
    //! ``mark_output`` nodes (not every historical ``mark_input``).
    PhaseSnapshot seal_phase();

    //! Same with an explicit carried list (overrides automatic marks).
    PhaseSnapshot seal_phase(std::vector<TensorNode const *> carried);

    void reset_phase_seal_cursor();

    size_t phase_seal_cursor() const { return phase_seal_cursor_; }

    //! Drop all sealed ops and rewind the seal cursor to 0.
    //! Ingress ``SCATTER`` is not special: once sealed and executed, values
    //! live in ``mark_input`` tile payloads. Unsealed ops past the seal
    //! cursor are always preserved (next phase recorded during a prior
    //! async ``run()``).
    void drop_all_ops();

    //! TensorNodes with no ``TensorRef`` and not referenced by remaining
    //! ops. Call after ``wait()`` plus ``drop_all_ops()`` once TileGraph
    //! ops are empty. Leaves holes in ``data_``; NodeIds stay.
    std::vector<TensorNode *> collect_dead_data_nodes() const;

    //! Unlink each node from its axis groups (O(ndim) swap-remove) and
    //! ``data_[id].reset()`` (O(1) hole). NodeIds are not reused or packed.
    //! Caller must drop TileGraph / Runtime maps first.
    void destroy_data_nodes(std::vector<TensorNode *> const &nodes);

    void note_axis_group(AxisDescriptor *group);
    void drop_axis_group(AxisDescriptor *group);

  private:
    friend void merge_axis(std::shared_ptr<AxisDescriptor> &,
        std::shared_ptr<AxisDescriptor> &);

    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> data_;
    std::vector<std::shared_ptr<TensorGraph::OpNode>> ops_;
    //! Distinct ``AxisDescriptor`` objects with at least one live member.
    std::unordered_set<AxisDescriptor *> axis_groups_;

    NodeId next_data_id_ = 0;
    NodeId next_op_id_ = 0;
    size_t phase_seal_cursor_ = 0;
};

//! One sealed slice of a ``TensorGraph``, ready for optional transforms and
//! lowering. ``tensor_graph`` points at the owning session graph; ``snapshot``
//! indexes ``tensor_graph->ops()``.
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

} // namespace nntile
