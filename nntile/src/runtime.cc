#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/runtime.cc
 * Runtime implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/runtime.hh"

#include "nntile/core/execution_schedule.hh"
#include "nntile/core/execution_worker.hh"

// TileGraph::get_tensor_descriptor is inline in graph.hh; this TU must see
// the definition when calling it on const TileGraph&.
#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor/graph_data_node.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tile/graph.hh"
#include "nntile/tile/graph_data_node.hh"
#include "nntile/tile/graph_op_node.hh"
#include "nntile/core/tile.hh"
#include "nntile/tile/lowering_context.hh"

#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nntile
{

namespace
{

template <typename T>
void allocate_tile_and_register(const TileGraph::TileNode *node,
    const std::vector<Index> &shape,
    std::map<const TileGraph::TileNode *, std::shared_ptr<void>> &tile_map)
{
    auto t = std::make_shared<nntile::core::Tile<T>>(shape);
    tile_map[node] = t;
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
    tile_adoption_.clear();

    execution_order_.clear();
    execution_order_.reserve(graph_.ops().size());
    for (const auto &op : graph_.ops())
    {
        execution_order_.push_back(op);
    }

    eliminate_dead_ops();

    execution_schedule_ = ExecutionSchedule{};

    compiled_ = true;
}

void Runtime::mark_initialized(TensorGraph::TensorNode const *tensor)
{
    if (tensor != nullptr)
    {
        init_state_[tensor] = true;
    }
}

bool Runtime::tensor_requires_init_at_execute(
    TileGraph::TensorDescriptor const &desc) const
{
    if (desc.source_node == nullptr)
    {
        return false;
    }
    if (!tile_bind_detail::tensor_desc_has_input_tile(desc))
    {
        return false;
    }
    std::unordered_set<const TileGraph::TileNode *> produced;
    for (const auto &op : execution_order_)
    {
        for (const auto *out : op->outputs())
        {
            if (out != nullptr)
            {
                produced.insert(out);
            }
        }
    }
    bool consumed = false;
    for (TileGraph::TileNode *tile : desc.tiles)
    {
        if (tile == nullptr || !tile->is_input())
        {
            continue;
        }
        if (produced.count(tile) != 0)
        {
            continue;
        }
        for (const auto &op : execution_order_)
        {
            for (const auto *in : op->inputs())
            {
                if (in == tile)
                {
                    consumed = true;
                    break;
                }
            }
            if (consumed)
            {
                break;
            }
        }
        if (consumed)
        {
            break;
        }
    }
    return consumed;
}

void Runtime::validate_initialized_inputs_at_compile()
{
    for (const auto &uptr : graph_.tensor_descriptors())
    {
        const TileGraph::TensorDescriptor &desc = *uptr;
        if (!tensor_requires_init_at_execute(desc))
        {
            continue;
        }
        if (!is_initialized(desc.source_node))
        {
            throw std::runtime_error(
                "Input is not initialized: " + desc.tensor_name);
        }
    }
}

void Runtime::export_initialized_tiles(
    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> &out) const
{
    out.clear();
    for (const auto &[tensor, initialized] : init_state_)
    {
        if (!initialized || tensor == nullptr)
        {
            continue;
        }
        const TileGraph::TensorDescriptor *desc =
            graph_.get_tensor_descriptor(tensor);
        if (desc == nullptr)
        {
            continue;
        }
        std::vector<std::shared_ptr<void>> ptrs;
        ptrs.reserve(desc->tiles.size());
        for (TileGraph::TileNode *tile : desc->tiles)
        {
            auto it = tile_map_.find(tile);
            if (it == tile_map_.end())
            {
                ptrs.clear();
                break;
            }
            ptrs.push_back(it->second);
        }
        if (!ptrs.empty())
        {
            out[tensor] = std::move(ptrs);
        }
    }
}

void Runtime::stage_persisted_tiles(
    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> const &persisted,
    TensorNodeToTileMap const &tile_map)
{
    tile_adoption_.clear();
    for (const auto &[tensor, saved_ptrs] : persisted)
    {
        auto tm_it = tile_map.find(tensor);
        if (tm_it == tile_map.end())
        {
            continue;
        }
        const std::vector<TileGraph::TileNode *> &new_tiles = tm_it->second;
        if (new_tiles.size() != saved_ptrs.size())
        {
            continue;
        }
        for (size_t i = 0; i < new_tiles.size(); ++i)
        {
            if (new_tiles[i] != nullptr)
            {
                tile_adoption_[new_tiles[i]] = saved_ptrs[i];
            }
        }
    }
}

void Runtime::restore_persisted_init_state(
    std::unordered_map<TensorGraph::TensorNode const *, bool> const
        &persisted_init)
{
    for (const auto &[tensor, initialized] : persisted_init)
    {
        if (tensor != nullptr && initialized)
        {
            init_state_[tensor] = true;
        }
    }
}

ExecutionSchedule Runtime::generate_round_robin_execution_schedule() const
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::generate_round_robin_execution_schedule: "
            "call compile() first");
    }
    return nntile::generate_round_robin_execution_schedule(
        graph_, execution_order_);
}

ExecutionSchedule Runtime::generate_affinity_batch_execution_schedule() const
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::generate_affinity_batch_execution_schedule: "
            "call compile() first");
    }
    return nntile::generate_affinity_batch_execution_schedule(
        graph_, execution_order_);
}

void Runtime::set_execution_schedule(ExecutionSchedule schedule)
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::set_execution_schedule: call compile() first");
    }
    if (schedule.fingerprint.op_count != 0 ||
        !schedule.fingerprint.op_names.empty())
    {
        validate_execution_schedule_fingerprint(
            schedule.fingerprint, execution_order_);
    }
    else if (schedule.ops.size() == execution_order_.size())
    {
        schedule.fingerprint =
            make_execution_schedule_fingerprint(execution_order_);
    }

    if (schedule.ops.size() != execution_order_.size())
    {
        throw std::runtime_error(
            "Runtime::set_execution_schedule: ops size (" +
            std::to_string(schedule.ops.size()) +
            ") != compiled execution order (" +
            std::to_string(execution_order_.size()) + ")");
    }
    int const num_workers = sched::count_execution_workers();
    bool const cuda_workers = starpu_is_initialized() &&
        starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) > 0;
    if (schedule.num_workers <= 0)
    {
        schedule.num_workers = num_workers;
    }
    if (schedule.num_workers != num_workers)
    {
        throw std::runtime_error(
            "Runtime::set_execution_schedule: num_workers mismatch (json '" +
            std::to_string(schedule.num_workers) +
            "' vs runtime '" + std::to_string(num_workers) + "')");
    }
    if (schedule.use_cuda_workers != cuda_workers)
    {
        throw std::runtime_error(
            "Runtime::set_execution_schedule: worker_kind mismatch (json '" +
            std::string(schedule.use_cuda_workers ? "cuda" : "cpu") +
            "' vs runtime '" + (cuda_workers ? "cuda" : "cpu") + "')");
    }
    for (size_t i = 0; i < schedule.ops.size(); ++i)
    {
        if (schedule.ops[i].execution_index != i)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: ops[" +
                std::to_string(i) + "].index mismatch");
        }
        if (schedule.ops[i].op_name != execution_order_[i]->op_name())
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: ops[" +
                std::to_string(i) + "] op_name mismatch (json '" +
                schedule.ops[i].op_name + "' vs graph '" +
                execution_order_[i]->op_name() + "')");
        }
        int const w = schedule.ops[i].worker;
        if (w < 0 || w >= num_workers)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: ops[" +
                std::to_string(i) + "] worker " + std::to_string(w) +
                " out of range [0, " + std::to_string(num_workers) + ")");
        }
    }
    for (auto const &[tile, worker] : schedule.tile_virtual_worker)
    {
        if (worker < 0 || worker >= num_workers)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: tile '" + tile +
                "' virtual_worker " + std::to_string(worker) +
                " out of range [0, " + std::to_string(num_workers) + ")");
        }
    }
    execution_schedule_ = std::move(schedule);
}

void Runtime::load_execution_schedule(std::string const &path)
{
    set_execution_schedule(load_execution_schedule_json(path));
}

void Runtime::apply_execution_schedule_from_file(std::string const &path)
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::apply_execution_schedule_from_file: call compile() "
            "first");
    }
    if (!execution_schedule_file_cache_ ||
        execution_schedule_file_cache_path_ != path)
    {
        execution_schedule_file_cache_ = load_execution_schedule_json(path);
        execution_schedule_file_cache_path_ = path;
    }
    try
    {
        set_execution_schedule(*execution_schedule_file_cache_);
    }
    catch (...)
    {
        clear_execution_schedule_file_cache();
        throw;
    }
}

void Runtime::clear_execution_schedule_file_cache()
{
    execution_schedule_file_cache_.reset();
    execution_schedule_file_cache_path_.clear();
}

void Runtime::compile_with_round_robin_schedule()
{
    compile();
    set_execution_schedule(generate_round_robin_execution_schedule());
}

void Runtime::write_execution_schedule_json(std::string const &path) const
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::write_execution_schedule_json: graph not compiled");
    }
    if (!has_execution_schedule())
    {
        throw std::runtime_error(
            "Runtime::write_execution_schedule_json: no schedule set");
    }
    nntile::write_execution_schedule_json(execution_schedule_, path);
}

void Runtime::require_compiled() const
{
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::execute: graph not compiled");
    }
}

void Runtime::allocate_missing_tiles()
{
    for (const auto &node : graph_.tile_nodes())
    {
        const TileGraph::TileNode *tile_key = node.get();
        auto adopt_it = tile_adoption_.find(tile_key);
        if (adopt_it != tile_adoption_.end())
        {
            tile_map_[tile_key] = adopt_it->second;
            continue;
        }
        if (tile_map_.count(tile_key) != 0)
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
    require_compiled();
    if (op_begin > op_end || op_end > execution_order_.size())
    {
        throw std::out_of_range("Runtime::execute_range: bad range");
    }
    bool const use_static_schedule = has_execution_schedule();
    for (size_t i = op_begin; i < op_end; ++i)
    {
        if (use_static_schedule)
        {
            starpu_worker_hint_ = sched::starpu_worker_id_for_scheduled_op(
                execution_schedule_.worker_for_op(i),
                execution_schedule_.use_cuda_workers,
                execution_order_[i]->op_name());
        }
        else
        {
            starpu_worker_hint_ = -1;
        }
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
    require_compiled();
    validate_initialized_inputs_at_compile();
    bool const use_static_schedule = has_execution_schedule();
    for (size_t i = 0; i < execution_order_.size(); ++i)
    {
        if (use_static_schedule)
        {
            starpu_worker_hint_ = sched::starpu_worker_id_for_scheduled_op(
                execution_schedule_.worker_for_op(i),
                execution_schedule_.use_cuda_workers,
                execution_order_[i]->op_name());
        }
        else
        {
            starpu_worker_hint_ = -1;
        }
        execution_order_[i]->execute(*this);
        // Global sync between ops (revisit when last-use invalidation
        // returns).
        starpu_task_wait_for_all();
    }
}

void Runtime::wait() { starpu_task_wait_for_all(); }

} // namespace nntile
