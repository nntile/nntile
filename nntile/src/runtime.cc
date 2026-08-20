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
#include "nntile/starpu/sync_defer.hh"

// TileGraph::get_tensor_descriptor is inline in graph.hh; this TU must see
// the definition when calling it on const TileGraph&.
#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor/graph_data_node.hh"
#include "nntile/tensor/tensor_ref.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tile/graph.hh"
#include "nntile/tile/graph_data_node.hh"
#include "nntile/tile/graph_op_node.hh"
#include "nntile/core/tile.hh"
#include "nntile/dtype.hh"
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

//! True if the tile's logical tensor still has a live ``TensorRef``.
bool tile_logical_is_live(TileGraph::TileNode const *tile)
{
    if (tile == nullptr)
    {
        return false;
    }
    auto *mutable_tile = const_cast<TileGraph::TileNode *>(tile);
    TileGraph::TensorDescriptor const *desc =
        mutable_tile->tensor_descriptor();
    return desc != nullptr && tensor_ref_is_live(desc->source_node);
}

//! Keep the logical tensor flag in sync with its tiles (O(tiles)).
void sync_logical_starpu_flag(TileGraph::TileNode *tile)
{
    if (tile == nullptr)
    {
        return;
    }
    TileGraph::TensorDescriptor const *desc =
        tile->tensor_descriptor();
    if (desc == nullptr || desc->source_node == nullptr)
    {
        return;
    }
    bool any = false;
    for (TileGraph::TileNode *t : desc->tiles)
    {
        if (t != nullptr && t->is_starpu_registered())
        {
            any = true;
            break;
        }
    }
    auto *src = const_cast<TensorGraph::TensorNode *>(
        desc->source_node);
    if (any)
    {
        src->note_starpu_registered();
    }
    else
    {
        src->note_starpu_unregistered();
    }
}

template <typename T>
void allocate_tile_and_register(
    TileGraph::TileNode *node, const std::vector<Index> &shape)
{
    auto t = std::make_shared<nntile::core::Tile<T>>(shape);
    node->set_payload(std::move(t));
    sync_logical_starpu_flag(node);
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
    const auto &graph_ops = graph_.ops();
    // Prefer appending only newly lowered ops. After a full wait(),
    // torch_nntile may clear TileGraph ops + reset compiled_graph_op_count_
    // via drop_fully_executed_history(); the size mismatch branch below
    // handles that. Keep already-executed prefix for Runtime::execute().
    if (compiled_graph_op_count_ > graph_ops.size())
    {
        compiled_graph_op_count_ = 0;
        compiled_tile_node_count_ = 0;
        execution_order_.clear();
        executed_op_end_ = 0;
    }
    if (compiled_graph_op_count_ < graph_ops.size())
    {
        execution_order_.reserve(
            execution_order_.size() +
            (graph_ops.size() - compiled_graph_op_count_));
        for (size_t i = compiled_graph_op_count_; i < graph_ops.size(); ++i)
        {
            execution_order_.push_back(graph_ops[i]);
        }
        compiled_graph_op_count_ = graph_ops.size();
    }

    eliminate_dead_ops();
    allocate_missing_tiles();
    tile_adoption_.clear();

    execution_schedule_ = ExecutionSchedule{};

    compiled_ = true;
}

void Runtime::invalidate_logical_tiles(
    TensorGraph::TensorNode const *logical)
{
    if (logical == nullptr)
    {
        return;
    }
    // Persistence is driven by marks: still-marked tensors stay allocated.
    if (tensor_ref_is_live(logical))
    {
        return;
    }
    const TileGraph::TensorDescriptor *desc =
        graph_.get_tensor_descriptor(logical);
    if (desc == nullptr)
    {
        return;
    }
    // Clear tile association; reclaim is driven by TensorRef lifetime.
    for (TileGraph::TileNode *tile : desc->tiles)
    {
        if (tile == nullptr)
        {
            continue;
        }
    }

    std::vector<const TileGraph::TileNode *> to_release;
    to_release.reserve(desc->tiles.size());
    for (TileGraph::TileNode *tile : desc->tiles)
    {
        if (tile == nullptr)
        {
            continue;
        }
        if (tile->has_payload() && tile->is_starpu_registered())
        {
            to_release.push_back(tile);
        }
    }
    if (to_release.empty())
    {
        init_state_.erase(logical);
        return;
    }
    // Async reclaim only. StarPU orders invalidate_submit /
    // unregister_submit after the handle's last submitted use, so this is
    // safe during overlapping compile/run phases (no wait_for_all).
    for (const TileGraph::TileNode *tile : to_release)
    {
        if (tile == nullptr || !tile->is_starpu_registered()
            || !tile->has_payload())
        {
            continue;
        }
        auto payload = tile->payload();
        invalidate_tile_buffer(tile, payload);
        auto *mut = const_cast<TileGraph::TileNode *>(tile);
        mut->clear_payload();
        sync_logical_starpu_flag(mut);
    }
    init_state_.erase(logical);
}

void Runtime::invalidate_tile(TileGraph::TileNode *tile)
{
    if (tile == nullptr || !tile->is_starpu_registered()
        || !tile->has_payload())
    {
        return;
    }
    auto payload = tile->payload();
    invalidate_tile_buffer(tile, payload);
    tile->clear_payload();
    sync_logical_starpu_flag(tile);
}

void Runtime::unregister_tile(TileGraph::TileNode *tile)
{
    if (tile == nullptr || !tile->is_starpu_registered())
    {
        return;
    }
    if (tile->has_payload())
    {
        auto payload = tile->payload();
        unregister_tile_buffer(tile, payload);
    }
    tile->clear_payload();
    sync_logical_starpu_flag(tile);
}

void Runtime::mark_initialized(TensorGraph::TensorNode const *tensor)
{
    if (tensor != nullptr)
    {
        init_state_[tensor] = true;
    }
}

void Runtime::forget_logical(TensorGraph::TensorNode const *tensor)
{
    if (tensor == nullptr)
    {
        return;
    }
    init_state_.erase(tensor);
    TileGraph::TensorDescriptor const *desc =
        graph_.get_tensor_descriptor(tensor);
    if (desc == nullptr)
    {
        return;
    }
    for (TileGraph::TileNode *tile : desc->tiles)
    {
        if (tile == nullptr)
        {
            continue;
        }
        tile_adoption_.erase(tile);
        live_tile_nodes_.erase(tile);
    }
}

bool Runtime::tensor_requires_init_at_execute(
    TileGraph::TensorDescriptor const &desc) const
{
    if (desc.source_node == nullptr)
    {
        return false;
    }
    if (!tile_bind_detail::tensor_desc_logical_is_live(desc))
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
        if (tile == nullptr || !tile_logical_is_live(tile))
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
        if (!uptr)
        {
            continue;
        }
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
            if (tile == nullptr || !tile->has_payload())
            {
                ptrs.clear();
                break;
            }
            ptrs.push_back(tile->payload());
        }
        if (!ptrs.empty())
        {
            out[tensor] = std::move(ptrs);
        }
    }
}

void Runtime::export_all_tiles(
    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> &out) const
{
    out.clear();
    for (const auto &uptr : graph_.tensor_descriptors())
    {
        if (!uptr)
        {
            continue;
        }
        const TileGraph::TensorDescriptor &desc = *uptr;
        if (desc.source_node == nullptr)
        {
            continue;
        }
        std::vector<std::shared_ptr<void>> ptrs;
        ptrs.reserve(desc.tiles.size());
        for (TileGraph::TileNode *tile : desc.tiles)
        {
            if (tile == nullptr || !tile->has_payload())
            {
                ptrs.clear();
                break;
            }
            ptrs.push_back(tile->payload());
        }
        if (!ptrs.empty())
        {
            out[desc.source_node] = std::move(ptrs);
        }
    }
}

std::vector<TensorGraph::TensorNode const *> Runtime::stage_persisted_tiles(
    std::unordered_map<TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> const &persisted,
    TensorNodeToTileMap const &tile_map)
{
    std::vector<TensorGraph::TensorNode const *> adopted;
    tile_adoption_.clear();
    for (const auto &[tensor, saved_ptrs] : persisted)
    {
        auto const *new_tiles_ptr = tile_map.try_get(tensor);
        if (new_tiles_ptr == nullptr)
        {
            continue;
        }
        const std::vector<TileGraph::TileNode *> &new_tiles = *new_tiles_ptr;
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
        adopted.push_back(tensor);
    }
    return adopted;
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
    std::unordered_set<const TileGraph::TileNode *> needed_by_pending;
    const size_t n = execution_order_.size();
    const size_t begin =
        executed_op_end_ < n ? executed_op_end_ : n;
    for (size_t i = begin; i < n; ++i)
    {
        for (const auto *in : execution_order_[i]->inputs())
        {
            if (in != nullptr)
            {
                needed_by_pending.insert(in);
            }
        }
        for (const auto *out : execution_order_[i]->outputs())
        {
            if (out != nullptr)
            {
                needed_by_pending.insert(out);
            }
        }
    }

    auto try_allocate = [&](const TileGraph::TileNode *tile_key,
        bool require_live_or_needed)
    {
        if (tile_key == nullptr)
        {
            return;
        }
        // Skip dead temps that no pending op needs. Newly lowered tiles
        // (ingress staging lowered before any scatter is appended) pass
        // require_live_or_needed=false so they still get buffers.
        if (require_live_or_needed && !tile_logical_is_live(tile_key) &&
            needed_by_pending.count(tile_key) == 0)
        {
            return;
        }
        auto adopt_it = tile_adoption_.find(tile_key);
        if (adopt_it != tile_adoption_.end())
        {
            auto *node = const_cast<TileGraph::TileNode *>(tile_key);
            node->set_payload(adopt_it->second);
            sync_logical_starpu_flag(node);
            return;
        }
        if (tile_key->has_payload())
        {
            return;
        }
        // graph_.tile_nodes() owns the unique_ptr; tile_key is non-owning.
        DataType dtype = tile_key->dtype();
        std::vector<Index> shape = tile_key->shape();
        auto *node = const_cast<TileGraph::TileNode *>(tile_key);

        switch (dtype)
        {
        case DataType::FP32:
            allocate_tile_and_register<nntile::fp32_t>(node, shape);
            break;
        case DataType::FP32_FAST_TF32:
            allocate_tile_and_register<nntile::fp32_fast_tf32_t>(
                node, shape);
            break;
        case DataType::FP32_FAST_FP16:
            allocate_tile_and_register<nntile::fp32_fast_fp16_t>(
                node, shape);
            break;
        case DataType::FP32_FAST_BF16:
            allocate_tile_and_register<nntile::fp32_fast_bf16_t>(
                node, shape);
            break;
        case DataType::FP64:
            allocate_tile_and_register<nntile::fp64_t>(node, shape);
            break;
        case DataType::FP16:
            allocate_tile_and_register<nntile::fp16_t>(node, shape);
            break;
        case DataType::BF16:
            allocate_tile_and_register<nntile::bf16_t>(node, shape);
            break;
        case DataType::INT64:
            allocate_tile_and_register<nntile::int64_t>(node, shape);
            break;
        case DataType::BOOL:
            allocate_tile_and_register<nntile::bool_t>(node, shape);
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for tile allocation");
        }
    };

    // Only touch pending I/O, DCE-live tiles, and newly lowered tile nodes -
    // never scan the full historical tile_nodes() list (that made compile
    // O(session length)). New nodes cover ingress staging lowered before any
    // scatter op is appended to execution_order_.
    for (const auto *tile : needed_by_pending)
    {
        try_allocate(tile, true);
    }
    for (const auto *tile : live_tile_nodes_)
    {
        if (needed_by_pending.count(tile) != 0)
        {
            continue;
        }
        try_allocate(tile, true);
    }
    const auto &all_tiles = graph_.tile_nodes();
    if (compiled_tile_node_count_ > all_tiles.size())
    {
        compiled_tile_node_count_ = 0;
    }
    for (size_t i = compiled_tile_node_count_; i < all_tiles.size(); ++i)
    {
        try_allocate(all_tiles[i].get(), false);
    }
    compiled_tile_node_count_ = all_tiles.size();
}

void Runtime::execute_range(
    size_t op_begin,
    size_t op_end,
    bool submit_tasks)
{
    require_compiled();
    if (op_begin > op_end || op_end > execution_order_.size())
    {
        throw std::out_of_range("Runtime::execute_range: bad range");
    }
    // Submit only: core sync wrappers skip wait_for_all while deferred so
    // torch_nntile run() can return before StarPU finishes the phase.
    // Unmarked-temp reclaim is ordinary TILE_INVALIDATE ops in this stream
    // (appended at compile), not a side-channel flush.
    StarpuSyncDefer defer_waits;
    bool const use_static_schedule =
        submit_tasks && has_execution_schedule();
    for (size_t i = op_begin; i < op_end; ++i)
    {
        if (submit_tasks)
        {
            if (use_static_schedule)
            {
                starpu_worker_hint_ =
                    sched::starpu_worker_id_for_scheduled_op(
                        execution_schedule_.worker_for_op(i),
                        execution_schedule_.use_cuda_workers,
                        execution_order_[i]->op_name());
            }
            else
            {
                starpu_worker_hint_ = -1;
            }
            execution_order_[i]->execute(*this);
        }
    }
    if (op_end > executed_op_end_)
    {
        executed_op_end_ = op_end;
    }
}

void Runtime::execute()
{
    require_compiled();
    validate_initialized_inputs_at_compile();
    // Full execute always re-runs the post-DCE order from index 0. Reset the
    // incremental watermark and reallocate unmarked tiles that a prior
    // compile may have skipped (pending slice was empty after a previous run).
    executed_op_end_ = 0;
    allocate_missing_tiles();
    // Submit only - same contract as execute_range. Call wait() to join
    // StarPU. Last-consumer invalidate_submit runs inside execute_range.
    execute_range(0, execution_order_.size());
}

void Runtime::build_tile_last_consumer_map()
{
    // Pending suffix only: size dying lists and last-consumer scratch to
    // O(pending), never O(|execution_order_|). Absolute op indices are
    // recovered via tiles_dying_op_base_.
    const size_t n = execution_order_.size();
    const size_t begin =
        executed_op_end_ < n ? executed_op_end_ : n;
    tiles_dying_op_base_ = begin;
    tiles_dying_after_op_.clear();
    if (begin >= n)
    {
        return;
    }
    const size_t pending = n - begin;
    tiles_dying_after_op_.assign(pending, {});

    // Sparse last-consumer over tiles touched by the pending suffix only.
    // Dense last_op_by_id[max_id+1] was O(session tile nodes) every compile
    // because TileNode ids are monotonic and append-only.
    std::unordered_map<const TileGraph::TileNode *, size_t> last_op;
    last_op.reserve(pending * 4);
    for (size_t i = begin; i < n; ++i)
    {
        for (const auto *in : execution_order_[i]->inputs())
        {
            if (in != nullptr)
            {
                last_op[in] = i;
            }
        }
    }
    for (const auto &[tile, last] : last_op)
    {
        tiles_dying_after_op_[last - begin].push_back(tile);
    }
}

void Runtime::invalidate_tile_buffer(
    const TileGraph::TileNode *node,
    const std::shared_ptr<void> &tile_ptr)
{
    if (node == nullptr || tile_ptr == nullptr)
    {
        return;
    }
    switch (node->dtype())
    {
    case DataType::FP32:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp32_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::FP32_FAST_TF32:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_tf32_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::FP32_FAST_FP16:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_fp16_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::FP32_FAST_BF16:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_bf16_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::FP64:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp64_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::FP16:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp16_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::BF16:
        std::static_pointer_cast<nntile::core::Tile<nntile::bf16_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::INT64:
        std::static_pointer_cast<nntile::core::Tile<nntile::int64_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    case DataType::BOOL:
        std::static_pointer_cast<nntile::core::Tile<nntile::bool_t>>(tile_ptr)
            ->invalidate_submit();
        break;
    default:
        break;
    }
}

void Runtime::unregister_tile_buffer(
    const TileGraph::TileNode *node,
    const std::shared_ptr<void> &tile_ptr)
{
    if (node == nullptr || tile_ptr == nullptr)
    {
        return;
    }
    switch (node->dtype())
    {
    case DataType::FP32:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp32_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::FP32_FAST_TF32:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_tf32_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::FP32_FAST_FP16:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_fp16_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::FP32_FAST_BF16:
        std::static_pointer_cast<
            nntile::core::Tile<nntile::fp32_fast_bf16_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::FP64:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp64_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::FP16:
        std::static_pointer_cast<nntile::core::Tile<nntile::fp16_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::BF16:
        std::static_pointer_cast<nntile::core::Tile<nntile::bf16_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::INT64:
        std::static_pointer_cast<nntile::core::Tile<nntile::int64_t>>(tile_ptr)
            ->unregister_submit();
        break;
    case DataType::BOOL:
        std::static_pointer_cast<nntile::core::Tile<nntile::bool_t>>(tile_ptr)
            ->unregister_submit();
        break;
    default:
        break;
    }
}

void Runtime::release_dead_tiles_after_op(size_t op_idx)
{
    queue_dead_tiles_after_op(op_idx);
    if (g_starpu_sync_defer_depth == 0)
    {
        starpu_task_wait_for_all_counted();
        flush_queued_dead_tiles();
    }
}

void Runtime::queue_dead_tiles_after_op(size_t op_idx)
{
    if (op_idx < tiles_dying_op_base_)
    {
        return;
    }
    size_t const local = op_idx - tiles_dying_op_base_;
    if (local >= tiles_dying_after_op_.size())
    {
        return;
    }
    for (const TileGraph::TileNode *tile : tiles_dying_after_op_[local])
    {
        // Skip tensors that still have a live TensorRef.
        if (tile == nullptr || tile_logical_is_live(tile))
        {
            continue;
        }
        queued_dead_tiles_.push_back(tile);
    }
}

void Runtime::flush_queued_dead_tiles()
{
    if (queued_dead_tiles_.empty())
    {
        return;
    }
    for (const TileGraph::TileNode *tile : queued_dead_tiles_)
    {
        if (tile == nullptr || !tile->is_starpu_registered()
            || !tile->has_payload())
        {
            continue;
        }
        auto payload = tile->payload();
        invalidate_tile_buffer(tile, payload);
        auto *mut = const_cast<TileGraph::TileNode *>(tile);
        mut->clear_payload();
        sync_logical_starpu_flag(mut);
    }
    queued_dead_tiles_.clear();
}

void Runtime::eliminate_dead_ops()
{
    live_tile_nodes_.clear();
    const size_t n = execution_order_.size();
    if (n == 0)
    {
        return;
    }
    // Already-executed prefix is immutable for incremental sessions; DCE only
    // the pending suffix so compile stays O(new phase), not O(history).
    const size_t pending_begin =
        executed_op_end_ < n ? executed_op_end_ : n;
    if (pending_begin >= n)
    {
        return;
    }

    using TNode = const TileGraph::TileNode *;
    std::unordered_map<TNode, std::unordered_set<size_t>> producer;
    std::unordered_map<TNode, std::unordered_set<size_t>> consumer;
    std::unordered_set<TNode> consumed;

    for (size_t i = pending_begin; i < n; ++i)
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
    // Seed from tiles whose logical still has a live TensorRef.
    for (size_t i = pending_begin; i < n; ++i)
    {
        const auto &op = execution_order_[i];
        for (const auto *out : op->outputs())
        {
            if (out != nullptr && tile_logical_is_live(out))
            {
                live_data.insert(out);
            }
        }
        for (const auto *in : op->inputs())
        {
            if (in != nullptr && tile_logical_is_live(in))
            {
                live_data.insert(in);
            }
        }
    }

    const bool any_live_output = !live_data.empty();
    if (!any_live_output)
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

    // Keep the executed prefix in place. Rebuilding the full vector every
    // compile recopied O(history) shared_ptrs and made step time grow.
    if (live_ops.size() == n - pending_begin)
    {
        live_tile_nodes_ = std::move(live_data);
        return;
    }
    size_t write = pending_begin;
    for (size_t i = pending_begin; i < n; ++i)
    {
        if (live_ops.count(i) == 0)
        {
            continue;
        }
        if (write != i)
        {
            execution_order_[write] = std::move(execution_order_[i]);
        }
        ++write;
    }
    execution_order_.resize(write);
    live_tile_nodes_ = std::move(live_data);
}

void Runtime::wait()
{
    starpu_task_wait_for_all_counted();
}

bool Runtime::drop_fully_executed_history()
{
    // Only safe when every compiled op has already run. Partial clears would
    // desync compiled_graph_op_count_ from TileGraph::ops() and force a
    // full re-append on the next compile().
    if (executed_op_end_ != execution_order_.size())
    {
        return false;
    }
    execution_order_.clear();
    executed_op_end_ = 0;
    compiled_graph_op_count_ = 0;
    live_tile_nodes_.clear();
    execution_schedule_ = ExecutionSchedule{};
    // Keep compiled_tile_node_count_: tile node slots persist (GC leaves
    // holes; new nodes still append). Dead TileNode IR is destroyed from
    // torch_nntile after wait + drop_fully_executed_history.
    return true;
}

} // namespace nntile
