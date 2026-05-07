/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/graph_runtime.cc
 * TileGraphExecutor implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/graph_runtime.hh"

// TileGraph::get_tensor_descriptor is inline in graph.hh; this TU must see
// the definition when calling it on const TileGraph&.
#include "nntile/graph/tile/graph.hh"

#include <cstring>
#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tile/graph_data_node.hh"
#include "nntile/graph/tile/graph_op_node.hh"
#include "nntile/graph/tensor/graph_data_node.hh"
#include "nntile/tile/tile.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void allocate_tile_and_register(
    const TileGraph::TileNode* node,
    const std::vector<Index>& shape,
    std::map<std::string, std::shared_ptr<void>>& runtime_data,
    std::map<std::string, DataType>& data_dtypes,
    std::map<const TileGraph::TileNode*, std::shared_ptr<void>>& tile_map)
{
    auto t = std::make_shared<nntile::tile::Tile<T>>(shape);
    runtime_data[node->name()] = t;
    data_dtypes[node->name()] = node->dtype();
    tile_map[node] = t;
}

void apply_multitile_bind_hint_from_source(
    const TensorGraphTiling& tsch,
    const TileGraph::TensorDescriptor& td,
    TileGraphExecutor& rt)
{
    const TensorGraph::TensorNode* src = td.source_node;
    if(src == nullptr)
    {
        return;
    }
    const std::vector<std::uint8_t>* hint = src->get_bind_hint();
    if(hint == nullptr)
    {
        return;
    }
    const TensorAxisLayout* lay = tsch.find(src);
    if(lay == nullptr)
    {
        throw std::runtime_error(
            "TileGraphExecutor::compile: missing tiling for multitile bind "
            "hint");
    }
    switch(td.dtype)
    {
        case DataType::FP32:
            tile_graph_layout_io::scatter_logical_tensor<float, nntile::fp32_t,
                                                         float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(float),
                rt);
            break;
        case DataType::FP32_FAST_TF32:
            tile_graph_layout_io::scatter_logical_tensor<
                float, nntile::fp32_fast_tf32_t, float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(nntile::fp32_fast_tf32_t),
                rt);
            break;
        case DataType::FP32_FAST_FP16:
            tile_graph_layout_io::scatter_logical_tensor<
                float, nntile::fp32_fast_fp16_t, float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(nntile::fp32_fast_fp16_t),
                rt);
            break;
        case DataType::FP32_FAST_BF16:
            tile_graph_layout_io::scatter_logical_tensor<
                float, nntile::fp32_fast_bf16_t, float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(nntile::fp32_fast_bf16_t),
                rt);
            break;
        case DataType::FP64:
            tile_graph_layout_io::scatter_logical_tensor<
                double, nntile::fp64_t, double>(
                *lay,
                td.tiles,
                reinterpret_cast<const double*>(hint->data()),
                hint->size() / sizeof(double),
                rt);
            break;
        case DataType::FP16:
            tile_graph_layout_io::scatter_logical_tensor<float, nntile::fp16_t,
                                                         float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(nntile::fp16_t),
                rt);
            break;
        case DataType::BF16:
            tile_graph_layout_io::scatter_logical_tensor<float, nntile::bf16_t,
                                                         float>(
                *lay,
                td.tiles,
                reinterpret_cast<const float*>(hint->data()),
                hint->size() / sizeof(nntile::bf16_t),
                rt);
            break;
        case DataType::INT64:
            tile_graph_layout_io::scatter_logical_tensor<
                std::int64_t, nntile::int64_t, std::int64_t>(
                *lay,
                td.tiles,
                reinterpret_cast<const std::int64_t*>(hint->data()),
                hint->size() / sizeof(std::int64_t),
                rt);
            break;
        case DataType::BOOL:
            tile_graph_layout_io::scatter_logical_tensor<bool, nntile::bool_t,
                                                         bool>(
                *lay,
                td.tiles,
                reinterpret_cast<const bool*>(hint->data()),
                hint->size() / sizeof(bool),
                rt);
            break;
        default:
            throw std::runtime_error(
                "apply_multitile_bind_hint_from_source: unsupported dtype");
    }
}

template<typename T>
void apply_bind_hint_impl(
    nntile::tile::Tile<T>& tile,
    const std::vector<std::uint8_t>& data)
{
    if(data.size() != static_cast<size_t>(tile.nelems) * sizeof(T))
    {
        throw std::runtime_error(
            "apply_bind_hint: data size mismatch");
    }
    auto tile_local = tile.acquire(STARPU_W);
    std::memcpy(tile_local.get_ptr(), data.data(), data.size());
    tile_local.release();
}

} // namespace

TileGraphExecutor::TileGraphExecutor(const TileGraph& graph)
    : graph_(graph)
{
}

DataType TileGraphExecutor::get_dtype(
    TensorGraph::TensorNode const* tensor) const
{
    const TileGraph::TensorDescriptor* d =
        graph_.get_tensor_descriptor(tensor);
    if(d != nullptr)
    {
        return d->dtype;
    }
    throw std::runtime_error(
        "TileGraphExecutor::get_dtype: unknown tensor data node");
}

void TileGraphExecutor::compile()
{
    allocate_missing_tiles();

    for(const auto& node : graph_.tile_nodes())
    {
        const auto* td = node->tensor_descriptor();
        if(td != nullptr && td->tiles.size() > static_cast<size_t>(1))
        {
            continue;
        }
        // Resolve bind hint: prefer source TensorNode, fall back to TileNode
        const std::vector<std::uint8_t>* hint = nullptr;
        if(td != nullptr && td->source_node != nullptr)
        {
            hint = td->source_node->get_bind_hint();
        }
        if(hint == nullptr)
        {
            hint = node->get_bind_hint();
        }
        if(hint == nullptr)
        {
            continue;
        }
        const std::string& name = node->name();
        DataType dtype = node->dtype();
        TileGraph::TileNode* tile_ptr = node.get();
        switch(dtype)
        {
            case DataType::FP32:
                apply_bind_hint_impl<nntile::fp32_t>(
                    get_tile<nntile::fp32_t>(tile_ptr), *hint);
                break;
            case DataType::FP32_FAST_TF32:
                apply_bind_hint_impl<nntile::fp32_fast_tf32_t>(
                    get_tile<nntile::fp32_fast_tf32_t>(tile_ptr), *hint);
                break;
            case DataType::FP32_FAST_FP16:
                apply_bind_hint_impl<nntile::fp32_fast_fp16_t>(
                    get_tile<nntile::fp32_fast_fp16_t>(tile_ptr), *hint);
                break;
            case DataType::FP32_FAST_BF16:
                apply_bind_hint_impl<nntile::fp32_fast_bf16_t>(
                    get_tile<nntile::fp32_fast_bf16_t>(tile_ptr), *hint);
                break;
            case DataType::FP64:
                apply_bind_hint_impl<nntile::fp64_t>(
                    get_tile<nntile::fp64_t>(tile_ptr), *hint);
                break;
            case DataType::FP16:
                apply_bind_hint_impl<nntile::fp16_t>(
                    get_tile<nntile::fp16_t>(tile_ptr), *hint);
                break;
            case DataType::BF16:
                apply_bind_hint_impl<nntile::bf16_t>(
                    get_tile<nntile::bf16_t>(tile_ptr), *hint);
                break;
            case DataType::INT64:
                apply_bind_hint_impl<nntile::int64_t>(
                    get_tile<nntile::int64_t>(tile_ptr), *hint);
                break;
            case DataType::BOOL:
                apply_bind_hint_impl<nntile::bool_t>(
                    get_tile<nntile::bool_t>(tile_ptr), *hint);
                break;
            default:
                throw std::runtime_error(
                    "apply_bind_hint: unsupported data type for " + name);
        }
    }

    if(const TensorGraphTiling* tsch = graph_.tiling_scheme())
    {
        for(const auto& uptr : graph_.tensor_descriptors())
        {
            const TileGraph::TensorDescriptor& td = *uptr;
            if(td.tiles.size() <= static_cast<size_t>(1))
            {
                continue;
            }
            if(td.source_node == nullptr ||
               td.source_node->get_bind_hint() == nullptr)
            {
                continue;
            }
            apply_multitile_bind_hint_from_source(*tsch, td, *this);
        }
    }

    execution_order_.clear();
    execution_order_.reserve(graph_.ops().size());
    for(const auto& op : graph_.ops())
    {
        execution_order_.push_back(op);
    }

    data_is_input_.clear();
    data_is_output_.clear();
    for(const auto& node : graph_.tile_nodes())
    {
        if(node->is_input())
        {
            data_is_input_.insert(node->name());
        }
        if(node->is_output())
        {
            data_is_output_.insert(node->name());
        }
    }

    for(const auto& uptr : graph_.tensor_descriptors())
    {
        const TileGraph::TensorDescriptor* td = uptr.get();
        bool any_in = false;
        bool any_out = false;
        for(TileGraph::TileNode* t : td->tiles)
        {
            if(t->is_input())
            {
                any_in = true;
            }
            if(t->is_output())
            {
                any_out = true;
            }
        }
        if(any_in)
        {
            data_is_input_.insert(td->tensor_name);
        }
        if(any_out)
        {
            data_is_output_.insert(td->tensor_name);
        }
    }

    eliminate_dead_ops();

    compiled_ = true;
}

void TileGraphExecutor::allocate_missing_tiles()
{
    for(const auto& node : graph_.tile_nodes())
    {
        if(tile_map_.count(node.get()) != 0)
        {
            continue;
        }
        DataType dtype = node->dtype();
        std::vector<Index> shape = node->shape();

        switch(dtype)
        {
            case DataType::FP32:
                allocate_tile_and_register<nntile::fp32_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::FP32_FAST_TF32:
                allocate_tile_and_register<nntile::fp32_fast_tf32_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::FP32_FAST_FP16:
                allocate_tile_and_register<nntile::fp32_fast_fp16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::FP32_FAST_BF16:
                allocate_tile_and_register<nntile::fp32_fast_bf16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::FP64:
                allocate_tile_and_register<nntile::fp64_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::FP16:
                allocate_tile_and_register<nntile::fp16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::BF16:
                allocate_tile_and_register<nntile::bf16_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::INT64:
                allocate_tile_and_register<nntile::int64_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            case DataType::BOOL:
                allocate_tile_and_register<nntile::bool_t>(
                    node.get(), shape, runtime_data_, data_dtypes_, tile_map_);
                break;
            default:
                throw std::runtime_error(
                    "Unsupported data type for tile allocation");
        }
    }
}

void TileGraphExecutor::execute_range(size_t op_begin, size_t op_end)
{
    if(!compiled_)
    {
        throw std::runtime_error(
            "TileGraphExecutor::execute_range: graph not compiled");
    }
    if(op_begin > op_end || op_end > execution_order_.size())
    {
        throw std::out_of_range(
            "TileGraphExecutor::execute_range: bad range");
    }
    for(size_t i = op_begin; i < op_end; ++i)
    {
        execution_order_[i]->execute(*this);
        starpu_task_wait_for_all();
    }
}

void TileGraphExecutor::eliminate_dead_ops()
{
    const size_t n = execution_order_.size();
    if(n == 0)
    {
        return;
    }

    std::unordered_map<std::string, std::unordered_set<size_t>> producer;
    std::unordered_map<std::string, std::unordered_set<size_t>> consumer;
    std::unordered_set<std::string> consumed;

    for(size_t i = 0; i < n; ++i)
    {
        const auto& op = execution_order_[i];
        for(const auto* out : op->outputs())
        {
            producer[out->name()].insert(i);
        }
        for(const auto* in : op->inputs())
        {
            consumed.insert(in->name());
            consumer[in->name()].insert(i);
        }
    }

    std::unordered_set<std::string> live_data(
        data_is_output_.begin(), data_is_output_.end());
    for(const auto& name : data_is_input_)
    {
        live_data.insert(name);
    }
    if(data_is_output_.empty())
    {
        for(const auto& p : producer)
        {
            if(consumed.count(p.first) == 0)
            {
                live_data.insert(p.first);
            }
        }
    }
    if(live_data.empty())
    {
        return;
    }

    std::set<size_t> live_ops;
    bool changed = true;
    while(changed)
    {
        changed = false;
        auto live_data_copy = live_data;
        for(const auto& t : live_data_copy)
        {
            auto prod_it = producer.find(t);
            if(prod_it != producer.end())
            {
                for(size_t op_idx : prod_it->second)
                {
                    if(live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        for(const auto* in : execution_order_[op_idx]->inputs())
                        {
                            if(live_data.insert(in->name()).second)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
            auto cons_it = consumer.find(t);
            if(cons_it != consumer.end())
            {
                for(size_t op_idx : cons_it->second)
                {
                    if(execution_order_[op_idx]->outputs().empty() &&
                       live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        for(const auto* in : execution_order_[op_idx]->inputs())
                        {
                            if(live_data.insert(in->name()).second)
                            {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<std::shared_ptr<OpNode>> filtered;
    filtered.reserve(live_ops.size());
    for(size_t i = 0; i < n; ++i)
    {
        if(live_ops.count(i))
        {
            filtered.push_back(execution_order_[i]);
        }
    }
    execution_order_ = std::move(filtered);
}

void TileGraphExecutor::execute()
{
    if(!compiled_)
    {
        throw std::runtime_error(
            "TileGraphExecutor::execute: graph not compiled");
    }
    for(size_t i = 0; i < execution_order_.size(); ++i)
    {
        execution_order_[i]->execute(*this);
        // Global sync between ops (revisit when last-use invalidation returns).
        starpu_task_wait_for_all();
    }
}

void TileGraphExecutor::wait()
{
    starpu_task_wait_for_all();
}

} // namespace nntile::graph
