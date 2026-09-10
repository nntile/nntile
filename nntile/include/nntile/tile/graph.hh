/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/graph.hh
 * TileGraph - graph operating on tiles. Purely symbolic; use
 * ``nntile::Runtime`` (``#include <nntile/runtime.hh>``) to run.
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/tile/graph_decl.hh>
#include <nntile/tile/graph_data_node.hh>
#include <nntile/tile/graph_op_node.hh>

namespace nntile
{

inline TileGraph::TileNode *TileGraph::data(
    std::vector<Index> shape, const std::string &name, DataType dtype)
{
    auto node = std::make_unique<TileNode>(
        next_data_id_, this, std::move(shape), dtype, name);
    ++next_data_id_;
    TileNode *node_ptr = node.get();

    data_.push_back(std::move(node));

    return node_ptr;
}

inline std::vector<std::string> TileGraph::data_names() const
{
    std::vector<std::string> names;
    names.reserve(data_.size());
    for (const auto &node : data_)
    {
        if (!node)
        {
            continue;
        }
        if (!node->name().empty())
        {
            names.push_back(node->name());
        }
    }
    return names;
}

inline void TileGraph::add_op(
    std::shared_ptr<OpNode> op_node, const std::string &name)
{
    for (const auto *input : op_node->inputs())
    {
        if (input->graph() != this)
        {
            throw std::invalid_argument("TileGraph::add_op: input data '" +
                                        input->name() +
                                        "' does not belong to this graph");
        }
    }

    for (const auto *output : op_node->outputs())
    {
        if (output->graph() != this)
        {
            throw std::invalid_argument("TileGraph::add_op: output data '" +
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

inline void TileGraph::clear_ops()
{
    ops_.clear();
    next_op_id_ = 0;
}

inline void TileGraph::erase_source_tensor(
    TensorGraph::TensorNode const *source)
{
    if (source == nullptr)
    {
        return;
    }
    auto const src_id = static_cast<size_t>(source->id());
    if (src_id >= tensors_.size() || !tensors_[src_id])
    {
        return;
    }
    TensorDescriptor *desc = tensors_[src_id].get();
    for (TileNode *tile : desc->tiles)
    {
        if (tile == nullptr)
        {
            continue;
        }
        auto const id = static_cast<size_t>(tile->id());
        if (id < data_.size() && data_[id].get() == tile)
        {
            data_[id].reset();
        }
    }
    tensors_[src_id].reset();
}

inline TileGraph::TensorDescriptor *TileGraph::add_tensor_descriptor(
    TensorDescriptor desc)
{
    if (desc.source_node == nullptr)
    {
        throw std::invalid_argument(
            "TileGraph::add_tensor_descriptor: source_node is required");
    }
    auto const id = static_cast<size_t>(desc.source_node->id());
    if (id >= tensors_.size())
    {
        tensors_.resize(id + 1);
    }
    if (tensors_[id])
    {
        throw std::invalid_argument(
            "TileGraph::add_tensor_descriptor: source already has a "
            "descriptor");
    }
    tensors_[id] = std::make_unique<TensorDescriptor>(std::move(desc));
    return tensors_[id].get();
}

inline TileGraph::TileNode *TileGraph::get_tile_node(const std::string &name)
{
    for (auto &node : data_)
    {
        if (!node)
        {
            continue;
        }
        if (node->name() == name)
        {
            return node.get();
        }
    }
    return nullptr;
}

inline const TileGraph::TileNode *TileGraph::get_tile_node(
    const std::string &name) const
{
    for (const auto &node : data_)
    {
        if (!node)
        {
            continue;
        }
        if (node->name() == name)
        {
            return node.get();
        }
    }
    return nullptr;
}

inline TileGraph::TensorDescriptor *TileGraph::get_tensor_descriptor(
    TensorGraph::TensorNode const *source)
{
    if (source == nullptr)
    {
        return nullptr;
    }
    auto const id = static_cast<size_t>(source->id());
    if (id >= tensors_.size())
    {
        return nullptr;
    }
    return tensors_[id].get();
}

inline const TileGraph::TensorDescriptor *TileGraph::get_tensor_descriptor(
    TensorGraph::TensorNode const *source) const
{
    if (source == nullptr)
    {
        return nullptr;
    }
    auto const id = static_cast<size_t>(source->id());
    if (id >= tensors_.size())
    {
        return nullptr;
    }
    return tensors_[id].get();
}

inline std::string TileGraph::to_string() const
{
    std::stringstream ss;
    ss << "TileGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", tiles=" << num_data() << ", ops=" << num_ops() << ")\n";

    if (!tensors_.empty())
    {
        ss << "Tensors:\n";
        for (const auto &td : tensors_)
        {
            if (!td)
            {
                continue;
            }
            ss << "  " << td->tensor_name << " shape=[";
            for (size_t i = 0; i < td->tensor_shape.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << td->tensor_shape[i];
            }
            ss << "] tile=[";
            for (size_t i = 0; i < td->tile_shape.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << td->tile_shape[i];
            }
            ss << "] grid=[";
            for (size_t i = 0; i < td->grid_shape.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << td->grid_shape[i];
            }
            ss << "] tiles=" << td->tiles.size() << "\n";
        }
    }

    ss << "Tiles:\n";
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

//! Mermaid TD graph: tiles as D*, ops as O* (same wiring pattern as
//! TensorGraph::to_mermaid). Tile labels include dtype, logical tensor
//! shape / tile / grid, axis names with tiling when source_node is set, and
//! grid coordinate + local tile shape.
inline std::string TileGraph::to_mermaid() const
{
    //! Bracketed index list for Mermaid labels, e.g. [1,2,3].
    auto index_list = [](const std::vector<Index> &v)
    {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i > 0)
            {
                s += ",";
            }
            s += std::to_string(static_cast<long long>(v[i]));
        }
        return s + "]";
    };

    std::stringstream ss;
    ss << "graph TD\n";

    for (const auto &node : data_)
    {
        const TileNode *tile = node.get();
        if (tile == nullptr)
        {
            continue;
        }
        std::string node_id = "D" + std::to_string(tile->id());
        std::string label = tile->name();
        if (label.empty())
        {
            label = "Tile" + std::to_string(tile->id());
        }

        label += "\\n" + dtype_to_string(tile->dtype());

        const TensorDescriptor *td = tile->tensor_descriptor();
        if (td != nullptr)
        {
            label += "\\n" + td->tensor_name + " full" +
                     index_list(td->tensor_shape);
            label += "\\n tile" + index_list(td->tile_shape) + " grid" +
                     index_list(td->grid_shape);
            const TensorGraph::TensorNode *src = td->source_node;
            if (src != nullptr)
            {
                // Same axis annotation style as TensorGraph::to_mermaid().
                std::string axes_str = "[";
                for (size_t i = 0; i < src->axes().size(); ++i)
                {
                    if (i > 0)
                    {
                        axes_str += ",";
                    }
                    const auto &ax = src->axes()[i];
                    if (!ax->name.empty())
                    {
                        axes_str += ax->name;
                    }
                    else
                    {
                        axes_str +=
                            std::to_string(static_cast<long long>(ax->extent));
                    }
                    if (ax->is_tiled())
                    {
                        axes_str += "/" + ax->tile_sizes_to_string();
                    }
                }
                axes_str += "]";
                label += "\\n" + axes_str;
            }
            if (!tile->tile_coord().empty())
            {
                label += "\\n@" + index_list(tile->tile_coord()) + " local" +
                         index_list(tile->shape());
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

    for (const auto &op : ops_)
    {
        std::string op_id = "O" + std::to_string(op->id());
        std::string label = op->op_name();
        if (!op->name().empty())
        {
            label += "\\n" + op->name();
        }
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

inline void TileGraph::rename_tile_node(TileNode *node, std::string new_name)
{
    if (node == nullptr || node->graph() != this)
    {
        throw std::invalid_argument(
            "TileGraph::rename_tile_node: invalid node");
    }
    if (new_name == node->name_)
    {
        return;
    }
    node->name_ = std::move(new_name);
}

inline TileGraph::TileNode *TileGraph::TileNode::set_name(std::string new_name)
{
    graph_->rename_tile_node(this, std::move(new_name));
    return this;
}

} // namespace nntile
