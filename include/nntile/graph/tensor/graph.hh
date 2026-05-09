/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph.hh
 * TensorGraph - graph operating on tensors. Purely symbolic; lower to
 * TileGraph and use TileGraphExecutor (``TileGraph::Runtime``) for execution.
 * Autograd is handled by ``NNGraph``, not by this class.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <set>

// NNTile headers
#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/graph/tensor/graph_decl.hh>
#include <nntile/graph/tensor/graph_op_node.hh>

namespace nntile::graph
{

inline TensorGraph::TensorNode *TensorGraph::data(
    std::vector<Index> shape, DataType dtype)
{
    auto node = std::make_unique<TensorNode>(
        next_data_id_, this, std::move(shape), dtype, "");
    ++next_data_id_;
    TensorNode *node_ptr = node.get();

    data_.push_back(std::move(node));

    return node_ptr;
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

    for (const auto *output : op_node->outputs())
    {
        if (output->graph() != this)
        {
            throw std::invalid_argument("TensorGraph::add_op: output data '" +
                                        output->name() +
                                        "' does not belong to this graph");
        }
    }

    op_node->id_ = next_op_id_++;
    if (!name.empty())
    {
        op_node->set_name(name);
    }
    ops_.push_back(std::move(op_node));
}

inline TensorGraph::PhaseSnapshot TensorGraph::seal_phase()
{
    std::vector<TensorNode const *> carried;
    carried.reserve(data_.size());
    for (auto const &node : data_)
    {
        TensorNode const *t = node.get();
        if (t->is_input() || t->is_output())
        {
            carried.push_back(t);
        }
    }
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

inline void TensorGraph::reset_phase_seal_cursor() { phase_seal_cursor_ = 0; }

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
        names.push_back(node->name());
    }
    return names;
}

inline std::vector<AxisDescriptor *> TensorGraph::axis_groups() const
{
    std::set<AxisDescriptor *> seen;
    std::vector<AxisDescriptor *> result;
    for (const auto &node : data_)
    {
        for (const auto &ax : node->axes())
        {
            if (seen.insert(ax.get()).second)
            {
                result.push_back(ax.get());
            }
        }
    }
    return result;
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
    ss << "TensorGraph(name='" << name_ << "', data=" << num_data()
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

} // namespace nntile::graph
