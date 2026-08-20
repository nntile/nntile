/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/graph.hh
 * TensorGraph - graph operating on tensors. Purely symbolic; lower to
 * ``TileGraph`` and use ``nntile::Runtime`` (``#include
 * <nntile/runtime.hh>``) for execution.
 * Autograd is owned by PyTorch / libtorch_nntile, not by this class.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <algorithm>
#include <unordered_set>
#include <utility>

// NNTile headers
#include <nntile/tensor/graph_decl.hh>
#include <nntile/tensor/graph_data_node.hh>
#include <nntile/tensor/graph_op_node.hh>
#include <nntile/tensor/tensor_ref.hh>

namespace nntile
{

inline TensorGraph::TensorNode *TensorGraph::emplace_data(
    std::vector<Index> shape, DataType dtype)
{
    auto node = std::make_unique<TensorNode>(
        next_data_id_, this, std::move(shape), dtype, "");
    ++next_data_id_;
    TensorNode *node_ptr = node.get();

    data_.push_back(std::move(node));
    return node_ptr;
}

inline TensorRef TensorGraph::data(
    std::vector<Index> shape, DataType dtype)
{
    return TensorRef::adopt(emplace_data(std::move(shape), dtype));
}

inline void TensorGraph::add_op(
    std::shared_ptr<OpNode> op_node, const std::string &name)
{
    for (const auto *input : op_node->inputs())
    {
        if (input->graph() != this)
        {
            throw std::invalid_argument("TensorGraph::add_op: input data '" +
                                        input->name() +
                                        "' does not belong to this graph");
        }
    }

    for (auto *output : op_node->outputs())
    {
        if (output->graph() != this)
        {
            throw std::invalid_argument("TensorGraph::add_op: output data '" +
                                        output->name() +
                                        "' does not belong to this graph");
        }
        output->note_produced();
        if (op_node->op_name() != "FILL")
        {
            output->clear_constant_value();
        }
    }

    op_node->id_ = next_op_id_++;
    if (!name.empty())
    {
        op_node->set_name(name);
    }
    ops_.push_back(std::move(op_node));
}

inline void TensorGraph::prepend_ops(
    std::vector<std::shared_ptr<TensorGraph::OpNode>> op_nodes)
{
    for (std::shared_ptr<TensorGraph::OpNode> &op_node : op_nodes)
    {
        if (op_node == nullptr)
        {
            throw std::invalid_argument(
                "TensorGraph::prepend_ops: op node must be non-null");
        }
        for (const TensorGraph::TensorNode *input : op_node->inputs())
        {
            if (input->graph() != this)
            {
                throw std::invalid_argument(
                    "TensorGraph::prepend_ops: input data '" +
                    input->name() + "' does not belong to this graph");
            }
        }
        for (TensorGraph::TensorNode *output : op_node->outputs())
        {
            if (output->graph() != this)
            {
                throw std::invalid_argument(
                    "TensorGraph::prepend_ops: output data '" +
                    output->name() + "' does not belong to this graph");
            }
            output->note_produced();
            if (op_node->op_name() != "FILL")
            {
                output->clear_constant_value();
            }
        }
        op_node->id_ = next_op_id_++;
    }
    ops_.insert(
        ops_.begin(),
        std::make_move_iterator(op_nodes.begin()),
        std::make_move_iterator(op_nodes.end()));
}

inline TensorGraph::PhaseSnapshot TensorGraph::seal_phase()
{
    // Carry only tensors touched by this phase's ops. Do not walk
    // marked_io_: every historical mark_input (e.g. preloaded batches)
    // lives there, and scanning it made seal O(session) each compile.
    // Live outputs used later appear as op inputs/outputs in that phase.
    std::unordered_set<TensorNode const *> touched;
    const size_t phase_ops = ops_.size() - phase_seal_cursor_;
    touched.reserve(phase_ops * 4 + 8);
    for (size_t i = phase_seal_cursor_; i < ops_.size(); ++i)
    {
        std::shared_ptr<OpNode> const &op = ops_[i];
        if (op == nullptr)
        {
            continue;
        }
        for (TensorNode const *in : op->inputs())
        {
            if (in != nullptr)
            {
                touched.insert(in);
            }
        }
        for (TensorNode *ot : op->outputs())
        {
            if (ot != nullptr)
            {
                touched.insert(ot);
            }
        }
    }
    std::vector<TensorNode const *> carried(
        touched.begin(),
        touched.end());
    // Stable order by node id so pending-compile equality checks compare
    // carried lists by index deterministically.
    std::sort(
        carried.begin(),
        carried.end(),
        [](TensorNode const *a, TensorNode const *b)
        {
            return a->id() < b->id();
        });
    return seal_phase(std::move(carried));
}

inline TensorGraph::PhaseSnapshot TensorGraph::seal_phase(
    std::vector<TensorNode const *> carried)
{
    PhaseSnapshot snap;
    snap.op_begin = phase_seal_cursor_;
    snap.op_end = ops_.size();
    snap.carried_tensors = std::move(carried);
    phase_seal_cursor_ = snap.op_end;
    return snap;
}

inline void TensorGraph::reset_phase_seal_cursor()
{
    phase_seal_cursor_ = 0;
}

inline void TensorGraph::drop_all_ops()
{
    // Drop every sealed op, including ingress SCATTER. Host-ingressed
    // values persist via mark_input + tile payloads after execute/wait;
    // retaining SCATTER edges made TensorGraph history O(#preloaded
    // batches). Unsealed ops past the seal cursor stay (next phase
    // recorded during a prior async run).
    // Dead TensorNode IR is destroyed separately via
    // ``destroy_data_nodes`` after wait (holes stay in ``data_``).
    if (phase_seal_cursor_ == 0)
    {
        return;
    }
    ops_.erase(
        ops_.begin(),
        ops_.begin() + static_cast<std::ptrdiff_t>(phase_seal_cursor_));
    phase_seal_cursor_ = 0;
}

inline size_t TensorGraph::num_live_data() const
{
    size_t n = 0;
    for (std::unique_ptr<TensorNode> const &up : data_)
    {
        if (up)
        {
            ++n;
        }
    }
    return n;
}

inline std::vector<TensorGraph::TensorNode *>
TensorGraph::collect_dead_data_nodes() const
{
    std::unordered_set<TensorNode const *> referenced;
    referenced.reserve(ops_.size() * 4 + 8);
    for (std::shared_ptr<OpNode> const &op : ops_)
    {
        if (op == nullptr)
        {
            continue;
        }
        for (TensorNode *in : op->inputs())
        {
            if (in != nullptr)
            {
                referenced.insert(in);
            }
        }
        for (TensorNode *ot : op->outputs())
        {
            if (ot != nullptr)
            {
                referenced.insert(ot);
            }
        }
    }
    std::vector<TensorNode *> dead;
    for (std::unique_ptr<TensorNode> const &up : data_)
    {
        TensorNode *t = up.get();
        if (t == nullptr)
        {
            continue;
        }
        if (tensor_ref_is_live(t))
        {
            continue;
        }
        // After wait() + empty TileGraph ops, leftover StarPU flags are
        // stale: reclaim already ran (INVALIDATE / UNREGISTER). Keeping
        // those nodes made live IR grow by one staging tensor per step.
        if (referenced.count(t) != 0)
        {
            continue;
        }
        dead.push_back(t);
    }
    return dead;
}

inline void TensorGraph::note_axis_group(AxisDescriptor *group)
{
    if (group != nullptr)
    {
        axis_groups_.insert(group);
    }
}

inline void TensorGraph::drop_axis_group(AxisDescriptor *group)
{
    if (group != nullptr)
    {
        axis_groups_.erase(group);
    }
}

inline void TensorGraph::destroy_data_nodes(
    std::vector<TensorNode *> const &nodes)
{
    for (TensorNode *t : nodes)
    {
        if (t == nullptr || t->graph() != this)
        {
            continue;
        }
        t->unlink_from_axis_groups();
        auto const id = static_cast<size_t>(t->id());
        if (id < data_.size() && data_[id].get() == t)
        {
            data_[id].reset();
        }
    }
}

inline void TensorGraph::rename_data_node(
    TensorNode *node, std::string new_name)
{
    if (node == nullptr || node->graph() != this)
    {
        throw std::invalid_argument(
            "TensorGraph::rename_data_node: invalid node");
    }
    if (new_name == node->name_)
    {
        return;
    }
    node->name_ = std::move(new_name);
}

inline TensorGraph::TensorNode *TensorGraph::TensorNode::set_name(
    std::string new_name)
{
    graph_->rename_data_node(this, std::move(new_name));
    return this;
}

inline std::vector<std::string> TensorGraph::data_names() const
{
    std::vector<std::string> names;
    names.reserve(data_.size());
    for (auto const &node : data_)
    {
        if (!node)
        {
            continue;
        }
        names.push_back(node->name());
    }
    return names;
}

inline std::vector<AxisDescriptor *> TensorGraph::axis_groups() const
{
    return {axis_groups_.begin(), axis_groups_.end()};
}

inline size_t TensorGraph::num_untiled_groups() const
{
    auto groups = axis_groups();
    size_t count = 0;
    for (const auto *g : groups)
    {
        if (!g->is_tiled())
        {
            ++count;
        }
    }
    return count;
}

inline std::string TensorGraph::to_string() const
{
    auto groups = axis_groups();
    size_t tiled = 0;
    for (const auto *g : groups)
    {
        if (g->is_tiled())
            ++tiled;
    }

    std::stringstream ss;
    ss << "TensorGraph(name='" << name_ << "', data=" << num_live_data()
       << "/" << num_data()
       << ", ops=" << num_ops() << ", axis_groups=" << groups.size()
       << ", tiled=" << tiled << "/" << groups.size() << ")\n";

    if (!groups.empty())
    {
        ss << "Axis groups:\n";
        for (const auto *g : groups)
        {
            ss << "  extent=" << g->extent;
            if (!g->name.empty())
            {
                ss << " name='" << g->name << "'";
            }
            if (g->is_tiled())
            {
                ss << " tile=" << g->tile_sizes_to_string();
            }
            ss << " members=" << g->members.size() << "\n";
        }
    }

    ss << "Data:\n";
    for (const auto &t : data_)
    {
        if (!t)
        {
            continue;
        }
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for (const auto &op : ops_)
    {
        ss << "  " << op->op_name() << "(id=" << op->id() << ")\n";
    }

    return ss.str();
}

inline std::string TensorGraph::to_mermaid() const
{
    std::stringstream ss;
    ss << "graph TD\n";

    for (const auto &node : data_)
    {
        if (!node)
        {
            continue;
        }
        std::string node_id = "D" + std::to_string(node->id());
        std::string label = node->name();
        if (label.empty())
            label = "Data" + std::to_string(node->id());

        std::string axes_str = "[";
        for (size_t i = 0; i < node->axes().size(); ++i)
        {
            if (i > 0)
                axes_str += ",";
            const auto &ax = node->axes()[i];
            if (!ax->name.empty())
            {
                axes_str += ax->name;
            }
            else
            {
                axes_str += std::to_string(ax->extent);
            }
            if (ax->is_tiled())
            {
                axes_str += "/" + ax->tile_sizes_to_string();
            }
        }
        axes_str += "]";
        label += "\\n" + dtype_to_string(node->dtype()) + "\\n" + axes_str;

        ss << "    " << node_id << "[\"" << label << "\"]\n";
    }

    for (const auto &op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        std::string label = op->op_name();
        if (!op->name().empty())
            label += "\\n" + op->name();
        ss << "    " << op_id << "{{\"" << label << "\"}}\n";
    }

    for (const auto &op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        for (const auto *input : op->inputs())
        {
            ss << "    D" << input->id() << " --> " << op_id << "\n";
        }
        for (const auto *output : op->outputs())
        {
            ss << "    " << op_id << " --> D" << output->id() << "\n";
        }
    }

    return ss.str();
}

} // namespace nntile
