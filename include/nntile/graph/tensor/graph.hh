/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph.hh
 * TensorGraph - graph operating on tensors. Purely symbolic; use CompiledGraph
 * for execution.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph_decl.hh>
#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/graph/tensor/graph_exec_ctx.hh>
#include <nntile/graph/tensor/graph_op_node.hh>

namespace nntile::graph
{

inline TensorGraph::TensorNode* TensorGraph::data(
    std::vector<Index> shape,
    const std::string& name,
    DataType dtype)
{
    if(data_by_name_.count(name) > 0)
    {
        throw std::invalid_argument(
            "TensorGraph::data: data '" + name + "' already exists");
    }

    auto node = std::make_unique<TensorNode>(
        next_data_id_,
        this,
        std::move(shape),
        dtype,
        name);
    ++next_data_id_;
    TensorNode* node_ptr = node.get();

    data_.push_back(std::move(node));
    data_by_name_[name] = node_ptr;

    return node_ptr;
}

inline void TensorGraph::add_op(std::shared_ptr<OpNode> op_node,
                                const std::string& name)
{
    for(const auto* input : op_node->inputs())
    {
        if(input->graph() != this)
        {
            throw std::invalid_argument(
                "TensorGraph::add_op: input data '" + input->name() +
                "' does not belong to this graph");
        }
    }

    for(const auto* output : op_node->outputs())
    {
        if(output->graph() != this)
        {
            throw std::invalid_argument(
                "TensorGraph::add_op: output data '" + output->name() +
                "' does not belong to this graph");
        }
    }

    op_node->id_ = next_op_id_++;
    if(!name.empty())
    {
        op_node->set_name(name);
    }
    ops_.push_back(std::move(op_node));
}

inline TensorGraph::TensorNode* TensorGraph::get_tensor_node(
    const std::string& name)
{
    auto it = data_by_name_.find(name);
    return it != data_by_name_.end() ? it->second : nullptr;
}

inline const TensorGraph::TensorNode* TensorGraph::get_tensor_node(
    const std::string& name) const
{
    auto it = data_by_name_.find(name);
    return it != data_by_name_.end() ? it->second : nullptr;
}

inline std::string TensorGraph::to_string() const
{
    std::stringstream ss;
    ss << "TensorGraph(name='" << name_ << "', data=" << num_data()
       << ", ops=" << num_ops() << ")\n";

    ss << "Data:\n";
    for(const auto& t : data_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for(const auto& op : ops_)
    {
        ss << "  " << op->op_name() << "(id=" << op->id() << ")\n";
    }

    return ss.str();
}

inline std::string TensorGraph::to_mermaid() const
{
    std::stringstream ss;
    ss << "graph TD\n";

    for(const auto& node : data_)
    {
        std::string node_id = "D" + std::to_string(node->id());
        std::string label = node->name();
        if(label.empty()) label = "Data" + std::to_string(node->id());

        std::string shape_str = "[";
        for(size_t i = 0; i < node->shape().size(); ++i)
        {
            if(i > 0) shape_str += ",";
            shape_str += std::to_string(node->shape()[i]);
        }
        shape_str += "]";
        label += "\\n" + dtype_to_string(node->dtype()) + "\\n" + shape_str;

        ss << "    " << node_id << "[\"" << label << "\"]\n";
    }

    for(const auto& op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        std::string label = op->op_name();
        if(!op->name().empty()) label += "\\n" + op->name();
        ss << "    " << op_id << "{{\"" << label << "\"}}\n";
    }

    for(const auto& op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        for(const auto* input : op->inputs())
        {
            ss << "    D" << input->id() << " --> " << op_id << "\n";
        }
        for(const auto* output : op->outputs())
        {
            ss << "    " << op_id << " --> D" << output->id() << "\n";
        }
    }

    return ss.str();
}

} // namespace nntile::graph
