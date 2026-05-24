/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/runtime.cc
 * Runtime implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/runtime.hh"

// TileGraph::get_tensor_descriptor is inline in graph.hh; this TU must see
// the definition when calling it on const TileGraph&.
#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor/graph_data_node.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tile/graph.hh"
#include "nntile/graph/tile/graph_data_node.hh"
#include "nntile/graph/tile/graph_op_node.hh"
#include "nntile/tile/tile.hh"

#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nntile::graph
{

namespace
{

template <typename T>
void allocate_tile_and_register(const TileGraph::TileNode *node,
    const std::vector<Index> &shape,
    std::map<const TileGraph::TileNode *, std::shared_ptr<void>> &tile_map)
{
    auto t = std::make_shared<nntile::tile::Tile<T>>(shape);
    tile_map[node] = t;
}

void apply_multitile_bind_hint_from_source(const TensorGraphTiling &tsch,
    const TileGraph::TensorDescriptor &td,
    Runtime &rt)
{
    const TensorGraph::TensorNode *src = td.source_node;
    if (src == nullptr)
    {
        return;
    }
    const std::vector<std::uint8_t> *hint = src->get_bind_hint();
    if (hint == nullptr)
    {
        return;
    }
    const TensorAxisLayout *lay = tsch.find(src);
    if (lay == nullptr)
    {
        throw std::runtime_error(
            "Runtime::compile: missing tiling for multitile bind "
            "hint");
    }
    switch (td.dtype)
    {
    case DataType::FP32:
        tile_graph_layout_io::
            scatter_logical_tensor<float, nntile::fp32_t, float>(*lay,
                td.tiles,
                reinterpret_cast<const float *>(hint->data()),
                hint->size() / sizeof(float),
                rt);
        break;
    case DataType::FP32_FAST_TF32:
        tile_graph_layout_io::scatter_logical_tensor<float,
            nntile::fp32_fast_tf32_t,
            float>(*lay,
            td.tiles,
            reinterpret_cast<const float *>(hint->data()),
            hint->size() / sizeof(nntile::fp32_fast_tf32_t),
            rt);
        break;
    case DataType::FP32_FAST_FP16:
        tile_graph_layout_io::scatter_logical_tensor<float,
            nntile::fp32_fast_fp16_t,
            float>(*lay,
            td.tiles,
            reinterpret_cast<const float *>(hint->data()),
            hint->size() / sizeof(nntile::fp32_fast_fp16_t),
            rt);
        break;
    case DataType::FP32_FAST_BF16:
        tile_graph_layout_io::scatter_logical_tensor<float,
            nntile::fp32_fast_bf16_t,
            float>(*lay,
            td.tiles,
            reinterpret_cast<const float *>(hint->data()),
            hint->size() / sizeof(nntile::fp32_fast_bf16_t),
            rt);
        break;
    case DataType::FP64:
        tile_graph_layout_io::
            scatter_logical_tensor<double, nntile::fp64_t, double>(*lay,
                td.tiles,
                reinterpret_cast<const double *>(hint->data()),
                hint->size() / sizeof(double),
                rt);
        break;
    case DataType::FP16:
        tile_graph_layout_io::
            scatter_logical_tensor<float, nntile::fp16_t, float>(*lay,
                td.tiles,
                reinterpret_cast<const float *>(hint->data()),
                hint->size() / sizeof(nntile::fp16_t),
                rt);
        break;
    case DataType::BF16:
        tile_graph_layout_io::
            scatter_logical_tensor<float, nntile::bf16_t, float>(*lay,
                td.tiles,
                reinterpret_cast<const float *>(hint->data()),
                hint->size() / sizeof(nntile::bf16_t),
                rt);
        break;
    case DataType::INT64:
        tile_graph_layout_io::scatter_logical_tensor<std::int64_t,
            nntile::int64_t,
            std::int64_t>(*lay,
            td.tiles,
            reinterpret_cast<const std::int64_t *>(hint->data()),
            hint->size() / sizeof(std::int64_t),
            rt);
        break;
    case DataType::BOOL:
        tile_graph_layout_io::
            scatter_logical_tensor<bool, nntile::bool_t, bool>(*lay,
                td.tiles,
                reinterpret_cast<const bool *>(hint->data()),
                hint->size() / sizeof(bool),
                rt);
        break;
    default:
        throw std::runtime_error(
            "apply_multitile_bind_hint_from_source: unsupported dtype");
    }
}

template <typename T>
void apply_bind_hint_impl(
    nntile::tile::Tile<T> &tile, const std::vector<std::uint8_t> &data)
{
    if (data.size() != static_cast<size_t>(tile.nelems) * sizeof(T))
    {
        throw std::runtime_error("apply_bind_hint: data size mismatch");
    }
    auto tile_local = tile.acquire(STARPU_W);
    std::memcpy(tile_local.get_ptr(), data.data(), data.size());
    tile_local.release();
}

//! Track both inputs and outputs when an op is needed: many kernels read
//! accumulator buffers listed only as outputs (in-place / incremental IR).
void insert_op_io_into_live(const TileGraph::OpNode &op,
    std::unordered_set<const TileGraph::TileNode *> &live,
    bool &changed)
{
    for (const auto *in : op.inputs())
    {
        if (in != nullptr && live.insert(in).second)
        {
            changed = true;
        }
    }
    for (const auto *out : op.outputs())
    {
        if (out != nullptr && live.insert(out).second)
        {
            changed = true;
        }
    }
}

} // namespace

Runtime::Runtime(const TileGraph &graph) : graph_(graph) {}

DataType Runtime::get_dtype(
    TensorGraph::TensorNode const *tensor) const
{
    const TileGraph::TensorDescriptor *d =
        graph_.get_tensor_descriptor(tensor);
    if (d != nullptr)
    {
        return d->dtype;
    }
    throw std::runtime_error(
        "Runtime::get_dtype: unknown tensor data node");
}

void Runtime::compile()
{
    allocate_missing_tiles();

    for (const auto &node : graph_.tile_nodes())
    {
        const auto *td = node->tensor_descriptor();
        if (td != nullptr && td->tiles.size() > static_cast<size_t>(1))
        {
            continue;
        }
        // Resolve bind hint: prefer source TensorNode, fall back to TileNode
        const std::vector<std::uint8_t> *hint = nullptr;
        if (td != nullptr && td->source_node != nullptr)
        {
            hint = td->source_node->get_bind_hint();
        }
        if (hint == nullptr)
        {
            hint = node->get_bind_hint();
        }
        if (hint == nullptr)
        {
            continue;
        }
        const std::string &name = node->name();
        DataType dtype = node->dtype();
        TileGraph::TileNode *tile_ptr = node.get();
        switch (dtype)
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

    if (const TensorGraphTiling *tsch = graph_.tiling_scheme())
    {
        for (const auto &uptr : graph_.tensor_descriptors())
        {
            const TileGraph::TensorDescriptor &td = *uptr;
            if (td.tiles.size() <= static_cast<size_t>(1))
            {
                continue;
            }
            if (td.source_node == nullptr ||
                td.source_node->get_bind_hint() == nullptr)
            {
                continue;
            }
            apply_multitile_bind_hint_from_source(*tsch, td, *this);
        }
    }

    execution_order_.clear();
    execution_order_.reserve(graph_.ops().size());
    for (const auto &op : graph_.ops())
    {
        execution_order_.push_back(op);
    }

    eliminate_dead_ops();

    compiled_ = true;
}

void Runtime::allocate_missing_tiles()
{
    for (const auto &node : graph_.tile_nodes())
    {
        if (tile_map_.count(node.get()) != 0)
        {
            continue;
        }
        DataType dtype = node->dtype();
        std::vector<Index> shape = node->shape();

        switch (dtype)
        {
        case DataType::FP32:
            allocate_tile_and_register<nntile::fp32_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::FP32_FAST_TF32:
            allocate_tile_and_register<nntile::fp32_fast_tf32_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::FP32_FAST_FP16:
            allocate_tile_and_register<nntile::fp32_fast_fp16_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::FP32_FAST_BF16:
            allocate_tile_and_register<nntile::fp32_fast_bf16_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::FP64:
            allocate_tile_and_register<nntile::fp64_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::FP16:
            allocate_tile_and_register<nntile::fp16_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::BF16:
            allocate_tile_and_register<nntile::bf16_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::INT64:
            allocate_tile_and_register<nntile::int64_t>(
                node.get(), shape, tile_map_);
            break;
        case DataType::BOOL:
            allocate_tile_and_register<nntile::bool_t>(
                node.get(), shape, tile_map_);
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for tile allocation");
        }
    }
}

void Runtime::execute_range(size_t op_begin, size_t op_end)
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::execute_range: graph not compiled");
    }
    if (op_begin > op_end || op_end > execution_order_.size())
    {
        throw std::out_of_range("Runtime::execute_range: bad range");
    }
    for (size_t i = op_begin; i < op_end; ++i)
    {
        execution_order_[i]->execute(*this);
        starpu_task_wait_for_all();
    }
}

void Runtime::eliminate_dead_ops()
{
    const size_t n = execution_order_.size();
    if (n == 0)
    {
        return;
    }

    using TNode = const TileGraph::TileNode *;
    std::unordered_map<TNode, std::unordered_set<size_t>> producer;
    std::unordered_map<TNode, std::unordered_set<size_t>> consumer;
    std::unordered_set<TNode> consumed;

    for (size_t i = 0; i < n; ++i)
    {
        const auto &op = execution_order_[i];
        for (const auto *out : op->outputs())
        {
            if (out != nullptr)
            {
                producer[out].insert(i);
            }
        }
        for (const auto *in : op->inputs())
        {
            if (in != nullptr)
            {
                consumed.insert(in);
                consumer[in].insert(i);
            }
        }
    }

    std::unordered_set<TNode> live_data;
    for (const auto &node : graph_.tile_nodes())
    {
        if (node->is_output())
        {
            live_data.insert(node.get());
        }
    }
    for (const auto &node : graph_.tile_nodes())
    {
        if (node->is_input())
        {
            live_data.insert(node.get());
        }
    }

    bool any_marked_output = false;
    for (const auto &node : graph_.tile_nodes())
    {
        if (node->is_output())
        {
            any_marked_output = true;
            break;
        }
    }
    if (!any_marked_output)
    {
        for (const auto &p : producer)
        {
            if (consumed.count(p.first) == 0)
            {
                live_data.insert(p.first);
            }
        }
    }
    if (live_data.empty())
    {
        return;
    }

    std::set<size_t> live_ops;
    bool changed = true;
    while (changed)
    {
        changed = false;
        auto live_data_copy = live_data;
        for (TNode t : live_data_copy)
        {
            auto prod_it = producer.find(t);
            if (prod_it != producer.end())
            {
                for (size_t op_idx : prod_it->second)
                {
                    if (live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        insert_op_io_into_live(
                            *execution_order_[op_idx], live_data, changed);
                    }
                }
            }
            // Any op that reads a live tile may be needed (sink ops have empty
            // outputs; others appear here when producer edges are
            // insufficient).
            auto cons_it = consumer.find(t);
            if (cons_it != consumer.end())
            {
                for (size_t op_idx : cons_it->second)
                {
                    if (live_ops.insert(op_idx).second)
                    {
                        changed = true;
                        insert_op_io_into_live(
                            *execution_order_[op_idx], live_data, changed);
                    }
                }
            }
        }
    }

    std::vector<std::shared_ptr<OpNode>> filtered;
    filtered.reserve(live_ops.size());
    for (size_t i = 0; i < n; ++i)
    {
        if (live_ops.count(i))
        {
            filtered.push_back(execution_order_[i]);
        }
    }
    execution_order_ = std::move(filtered);
}

void Runtime::execute()
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::execute: graph not compiled");
    }
    for (size_t i = 0; i < execution_order_.size(); ++i)
    {
        execution_order_[i]->execute(*this);
        // Global sync between ops (revisit when last-use invalidation
        // returns).
        starpu_task_wait_for_all();
    }
}

void Runtime::wait() { starpu_task_wait_for_all(); }

} // namespace nntile::graph
