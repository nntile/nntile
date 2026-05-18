/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/graph.hh
 * TileGraph - graph operating on tiles. Purely symbolic; use
 * TileGraph::Runtime for execution.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph_decl.hh>
#include <nntile/graph/tile/graph_data_node.hh>
#include <nntile/graph/tile/graph_runtime.hh>
#include <nntile/graph/tile/graph_op_node.hh>

namespace nntile::graph
{

inline TileGraph::TileNode* TileGraph::data(
    std::vector<Index> shape,
    const std::string& name,
    DataType dtype)
{
    if(data_by_name_.count(name) > 0)
    {
        throw std::invalid_argument(
            "TileGraph::data: data '" + name + "' already exists");
    }

    auto node = std::make_unique<TileNode>(
        next_data_id_,
        this,
        std::move(shape),
        dtype,
        name);
    ++next_data_id_;
    TileNode* node_ptr = node.get();

    data_.push_back(std::move(node));
    data_by_name_[name] = node_ptr;

    return node_ptr;
}

inline void TileGraph::add_op(std::shared_ptr<OpNode> op_node,
                              const std::string& name)
{
    for(const auto* input : op_node->inputs())
    {
        if(input->graph() != this)
        {
            throw std::invalid_argument(
                "TileGraph::add_op: input data '" + input->name() +
                "' does not belong to this graph");
        }
    }

    for(const auto* output : op_node->outputs())
    {
        if(output->graph() != this)
        {
            throw std::invalid_argument(
                "TileGraph::add_op: output data '" + output->name() +
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

inline TileGraph::TensorDescriptor* TileGraph::add_tensor_descriptor(
    TensorDescriptor desc)
{
    auto ptr = std::make_unique<TensorDescriptor>(std::move(desc));
    TensorDescriptor* raw = ptr.get();
    tensors_by_name_[raw->tensor_name] = raw;
    tensors_.push_back(std::move(ptr));
    return raw;
}

inline TileGraph::TileNode* TileGraph::get_tile_node(
    const std::string& name)
{
    auto it = data_by_name_.find(name);
    return it != data_by_name_.end() ? it->second : nullptr;
}

inline const TileGraph::TileNode* TileGraph::get_tile_node(
    const std::string& name) const
{
    auto it = data_by_name_.find(name);
    return it != data_by_name_.end() ? it->second : nullptr;
}

inline TileGraph::TensorDescriptor* TileGraph::get_tensor_descriptor(
    const std::string& tensor_name)
{
    auto it = tensors_by_name_.find(tensor_name);
    return it != tensors_by_name_.end() ? it->second : nullptr;
}

inline const TileGraph::TensorDescriptor* TileGraph::get_tensor_descriptor(
    const std::string& tensor_name) const
{
    auto it = tensors_by_name_.find(tensor_name);
    return it != tensors_by_name_.end() ? it->second : nullptr;
}

inline std::string TileGraph::to_string() const
{
    std::stringstream ss;
    ss << "TileGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", tiles=" << num_data() << ", ops=" << num_ops() << ")\n";

    if(!tensors_.empty())
    {
        ss << "Tensors:\n";
        for(const auto& td : tensors_)
        {
            ss << "  " << td->tensor_name << " shape=[";
            for(size_t i = 0; i < td->tensor_shape.size(); ++i)
            {
                if(i > 0) ss << ", ";
                ss << td->tensor_shape[i];
            }
            ss << "] tile=[";
            for(size_t i = 0; i < td->tile_shape.size(); ++i)
            {
                if(i > 0) ss << ", ";
                ss << td->tile_shape[i];
            }
            ss << "] grid=[";
            for(size_t i = 0; i < td->grid_shape.size(); ++i)
            {
                if(i > 0) ss << ", ";
                ss << td->grid_shape[i];
            }
            ss << "] tiles=" << td->tiles.size() << "\n";
        }
    }

    ss << "Tiles:\n";
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

//! Mermaid TD graph: tiles as D*, ops as O* (same wiring pattern as
//! TensorGraph::to_mermaid). Tile labels include dtype, logical tensor
//! shape / tile / grid, axis names with tiling when source_node is set, and
//! grid coordinate + local tile shape.
inline std::string TileGraph::to_mermaid() const
{
    //! Bracketed index list for Mermaid labels, e.g. [1,2,3].
    auto index_list = [](const std::vector<Index>& v) {
        std::string s = "[";
        for(size_t i = 0; i < v.size(); ++i)
        {
            if(i > 0)
            {
                s += ",";
            }
            s += std::to_string(static_cast<long long>(v[i]));
        }
        return s + "]";
    };

    std::stringstream ss;
    ss << "graph TD\n";

    for(const auto& node : data_)
    {
        const TileNode* tile = node.get();
        std::string node_id = "D" + std::to_string(tile->id());
        std::string label = tile->name();
        if(label.empty())
        {
            label = "Tile" + std::to_string(tile->id());
        }

        label += "\\n" + dtype_to_string(tile->dtype());

        const TensorDescriptor* td = tile->tensor_descriptor();
        if(td != nullptr)
        {
            label += "\\n" + td->tensor_name + " full" + index_list(td->tensor_shape);
            label += "\\n tile" + index_list(td->tile_shape) + " grid"
                + index_list(td->grid_shape);
            const TensorGraph::TensorNode* src = td->source_node;
            if(src != nullptr)
            {
                // Same axis annotation style as TensorGraph::to_mermaid().
                std::string axes_str = "[";
                for(size_t i = 0; i < src->axes().size(); ++i)
                {
                    if(i > 0)
                    {
                        axes_str += ",";
                    }
                    const auto& ax = src->axes()[i];
                    if(!ax->name.empty())
                    {
                        axes_str += ax->name;
                    }
                    else
                    {
                        axes_str +=
                            std::to_string(static_cast<long long>(ax->extent));
                    }
                    if(ax->is_tiled())
                    {
                        axes_str += "/" + ax->tile_sizes_to_string();
                    }
                }
                axes_str += "]";
                label += "\\n" + axes_str;
            }
            if(!tile->tile_coord().empty())
            {
                label += "\\n@" + index_list(tile->tile_coord()) + " local"
                    + index_list(tile->shape());
            }
            else
            {
                label += "\\n local" + index_list(tile->shape());
            }
        }
        else
        {
            label += "\\n local" + index_list(tile->shape());
        }

        ss << "    " << node_id << "[\"" << label << "\"]\n";
    }

    for(const auto& op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        std::string label = op->op_name();
        if(!op->name().empty())
        {
            label += "\\n" + op->name();
        }
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
