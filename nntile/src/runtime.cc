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
#ifdef NNTILE_USE_NNHAUL
#include "nntile/backend_workers.hh"
#include "nntile/nnhaul/chunk_gate.hh"
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/sync_defer.hh"
#endif

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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nntile
{

namespace
{

//! [C1 diag] env-gated compile sub-pass brackets (NNTILE_COMPILE_BREAKDOWN).
//! Pure observation: steady-clock reads + one stderr line per compile.
bool compile_breakdown_enabled()
{
    return env_flag_enabled("NNTILE_COMPILE_BREAKDOWN");
}

double breakdown_ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

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
    auto t = nntile::make_arena_shared<nntile::core::Tile<T>>(shape);
    node->set_payload(std::move(t));
    sync_logical_starpu_flag(node);
}

} // namespace

Runtime::Runtime(const TileGraph &graph) : graph_(graph) {}

Runtime::~Runtime()
{
#ifdef NNTILE_USE_NNHAUL
    if (!nnhaul_retained_payloads_.empty())
    {
        starpu_task_wait_for_all_counted();
        nnhaul_retained_payloads_.clear();
    }
#endif
}

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
        // The op history was dropped; rescan re-accumulates the flag.
        graph_has_torch_ops_ = false;
    }
    if (compiled_graph_op_count_ < graph_ops.size())
    {
        bool const cbrk = compile_breakdown_enabled();
        auto const t_append = std::chrono::steady_clock::now();
        size_t const new_ops = graph_ops.size() - compiled_graph_op_count_;
        execution_order_.reserve(execution_order_.size() + new_ops);
        for (size_t i = compiled_graph_op_count_; i < graph_ops.size(); ++i)
        {
            execution_order_.push_back(graph_ops[i]);
            if (!graph_has_torch_ops_ && graph_ops[i]->is_torch_dispatch())
            {
                graph_has_torch_ops_ = true;
            }
        }
        compiled_graph_op_count_ = graph_ops.size();
        if (cbrk)
        {
            std::fprintf(stderr,
                "[nntile-cbrk] rt new_ops=%zu eo=%zu append_ms=%.3f\n",
                new_ops, execution_order_.size(),
                breakdown_ms_since(t_append));
        }
    }
#ifdef NNTILE_USE_NNHAUL
    // Chunked tiles are for the classic stack only. Torch_dispatch ops
    // consume tiles through contiguous torch tensors and cannot read a
    // chunk chain, so the combination must be impossible to reach: if
    // this graph lowers a torch op while any chunked tile handle is
    // alive (every chunked tile registers itself in the stage registry
    // at allocation), fail LOUDLY at compile time - before execution
    // could hand a chunked buffer to a torch kernel. The gate flip below
    // only affects future allocations and would silently leave the old
    // chunked tiles consumable, hence the throw. NNHAUL_CHUNK_TILES=0
    // also forces unchunked handles (the one-class law has no such arm).
    if (graph_has_torch_ops_
        && ::nntile::haul::stage::chunked_handle_count() != 0)
    {
        throw std::runtime_error(
            "NNTile/NNHaul: this graph lowers torch_dispatch ops "
            "(TILE_TORCH_*), which cannot consume chunked NNHaul tiles, "
            "but chunked tile handles are already allocated in this "
            "session (the chunked-tile gate is on and chunked handles "
            "were allocated before this graph compiled). For classic "
            "sessions set NNHAUL_CHUNK_TILES=0 from process start; for "
            "torch_nntile sessions make sure NNHAUL_CHUNK_TILES is "
            "UNSET (the session policy then disables chunked tiles "
            "automatically) and that the libtorch_nntile.so in this "
            "process matches the installed libnntile.so.");
    }
    // Monotonic session law: a torch-op graph disables chunked tiles
    // for the REST of the session (future allocations only — the guard
    // above has already thrown if chunked handles were alive). Classic
    // graphs no longer FORCE chunking back on: a session that started
    // with chunking off (the torch_nntile session policy or
    // NNHAUL_CHUNK_TILES=0) keeps it off, while pure-classic sessions
    // are unaffected (the default is on and nothing ever turned it
    // off). The old unconditional form re-enabled chunking after every
    // classic graph's compile, which clobbered a session-level flip
    // made before any torch-op graph had compiled.
    if(graph_has_torch_ops_)
    {
        ::nntile::haul::set_tile_chunking(false);
    }
#endif

    auto const t_dce = std::chrono::steady_clock::now();
    // One flat SoA snapshot of the pending suffix, shared by the compile
    // passes below (DCE adjacency + mark, allocate_missing_tiles' needed
    // walk). Per-compile scratch only.
    PendingPhaseSnapshot snap = build_pending_phase_snapshot();
    if (compile_breakdown_enabled())
    {
        std::fprintf(stderr,
            "[nntile-cbrk] snap pending=%zu build_ms=%.3f "
            "in_edges=%zu out_edges=%zu\n",
            snap.in_off.size() - 1,
            breakdown_ms_since(t_dce), snap.in_edges.size(),
            snap.out_edges.size());
    }
    eliminate_dead_ops(&snap);
    last_compile_dce_s_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_dce).count();
    auto const t_alloc = std::chrono::steady_clock::now();
    allocate_missing_tiles(&snap);
    last_compile_alloc_s_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_alloc).count();
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
#ifdef NNTILE_USE_NNHAUL
        if (payload)
        {
            nnhaul_retained_payloads_.push_back(payload);
        }
#endif
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
#ifdef NNTILE_USE_NNHAUL
    if (payload)
    {
        nnhaul_retained_payloads_.push_back(payload);
    }
#endif
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
#ifdef NNTILE_USE_NNHAUL
        if (payload)
        {
            nnhaul_retained_payloads_.push_back(payload);
        }
#endif
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

void Runtime::install_default_execution_schedule_if_missing()
{
    if (has_execution_schedule())
    {
        return;
    }
    if (!compiled_)
    {
        throw std::runtime_error(
            "Runtime::install_default_execution_schedule_if_missing: "
            "call compile() first");
    }
    if (sched::execution_ncuda() > 0)
    {
        set_execution_schedule(
            generate_affinity_batch_execution_schedule());
    }
    else
    {
        set_execution_schedule(generate_round_robin_execution_schedule());
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
#ifdef NNTILE_USE_NNHAUL
    bool const cuda_workers = g_backend_ncuda > 0;
    if (schedule.workers.empty())
    {
        for (BackendWorker const &worker : backend_worker_list())
        {
            ScheduleWorkerInfo info;
            info.id = worker.id;
            info.kind = worker.kind;
            info.device = worker.device;
            schedule.workers.push_back(std::move(info));
        }
    }
#else
    bool const cuda_workers = starpu_is_initialized() &&
        starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) > 0;
#endif
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
#ifndef NNTILE_USE_NNHAUL
    if (schedule.use_cuda_workers != cuda_workers)
    {
        throw std::runtime_error(
            "Runtime::set_execution_schedule: worker_kind mismatch (json '" +
            std::string(schedule.use_cuda_workers ? "cuda" : "cpu") +
            "' vs runtime '" + (cuda_workers ? "cuda" : "cpu") + "')");
    }
#else
    (void)cuda_workers;
#endif
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
#ifdef NNTILE_USE_NNHAUL
        ScheduleWorkerInfo const *live = nullptr;
        for (ScheduleWorkerInfo const &info : schedule.workers)
        {
            if (info.id == w)
            {
                live = &info;
                break;
            }
        }
        if (live == nullptr)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: ops[" +
                std::to_string(i) + "] worker " + std::to_string(w) +
                " is not a live id");
        }
        bool const cuda_kind = live->kind == "cuda";
        bool const has_cuda =
            sched::tile_op_has_cuda_kernel(schedule.ops[i].op_name);
        if (cuda_kind && !has_cuda)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: ops[" +
                std::to_string(i) + "] worker kind has no kernel for " +
                schedule.ops[i].op_name);
        }
        if (g_schedule_kind_filter == ScheduleKindFilter::Cpu &&
            cuda_kind)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: CPU restriction");
        }
        if (g_schedule_kind_filter == ScheduleKindFilter::Cuda &&
            !cuda_kind)
        {
            throw std::runtime_error(
                "Runtime::set_execution_schedule: CUDA restriction");
        }
#endif
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

Runtime::PendingPhaseSnapshot Runtime::build_pending_phase_snapshot() const
{
    PendingPhaseSnapshot snap;
    const size_t n = execution_order_.size();
    const size_t begin = executed_op_end_ < n ? executed_op_end_ : n;
    snap.op_begin = begin;
    const size_t pending = n - begin;
    snap.in_off.assign(pending + 1, 0);
    snap.out_off.assign(pending + 1, 0);
    snap.is_unregister.assign(pending, 0);
    for (size_t local = 0; local < pending; ++local)
    {
        auto const &op = execution_order_[begin + local];
        auto const &ins = op->inputs();
        auto const &outs = op->outputs();
        for (const TileNode *t : ins)
        {
            snap.in_edges.push_back(t);
        }
        for (const TileNode *t : outs)
        {
            snap.out_edges.push_back(t);
        }
        snap.in_off[local + 1] =
            static_cast<std::uint32_t>(snap.in_edges.size());
        snap.out_off[local + 1] =
            static_cast<std::uint32_t>(snap.out_edges.size());
        snap.is_unregister[local] = op->is_tile_unregister() ? 1 : 0;
    }
    return snap;
}

void Runtime::allocate_missing_tiles(PendingPhaseSnapshot const *snap_in)
{
    bool const cbrk = compile_breakdown_enabled();
    auto const t_need = std::chrono::steady_clock::now();
    const size_t n = execution_order_.size();
    const size_t begin =
        executed_op_end_ < n ? executed_op_end_ : n;
    const size_t pending = n - begin;
    // Use the shared per-compile snapshot when it matches this pending
    // window; otherwise (Runtime::execute path) build a local one.
    PendingPhaseSnapshot own;
    PendingPhaseSnapshot const *snap = snap_in;
    if (snap == nullptr || snap->op_begin != begin
        || snap->in_off.size() != pending + 1)
    {
        own = build_pending_phase_snapshot();
        snap = &own;
    }
    // Dense needed-tile bitmap over the PENDING WINDOW's tile-id span.
    // TileNode ids are absolute slots in TileGraph::tile_nodes()
    // (session-monotonic), so the bitmap keeps absolute indices but is
    // sized to max tile id over the snapshot's op edges + 1: every edge
    // tile id is below that bound by construction, so indexing stays
    // absolute while the zero-init and scan cost scale with the pending
    // phase, not with session length. (A DCE-eliminated op's tiles are
    // never marked: DCE compacts execution_order_ before this pass and
    // the staleness check above rebuilds the snapshot from the post-DCE
    // order, which is what guarantees post-DCE correctness here.)
    size_t max_pending_id = 0;
    for (const TileNode *t : snap->in_edges)
    {
        if (t != nullptr)
        {
            size_t const id = static_cast<size_t>(t->id());
            if (id > max_pending_id)
            {
                max_pending_id = id;
            }
        }
    }
    for (const TileNode *t : snap->out_edges)
    {
        if (t != nullptr)
        {
            size_t const id = static_cast<size_t>(t->id());
            if (id > max_pending_id)
            {
                max_pending_id = id;
            }
        }
    }
    const size_t id_bound = max_pending_id + 1;
    std::vector<std::uint8_t> needed(id_bound, 0);
    for (size_t local = 0; local < pending; ++local)
    {
        for (std::uint32_t e = snap->in_off[local];
             e < snap->in_off[local + 1]; ++e)
        {
            const TileNode *t = snap->in_edges[e];
            if (t != nullptr)
            {
                needed[static_cast<size_t>(t->id())] = 1;
            }
        }
        for (std::uint32_t e = snap->out_off[local];
             e < snap->out_off[local + 1]; ++e)
        {
            const TileNode *t = snap->out_edges[e];
            if (t != nullptr)
            {
                needed[static_cast<size_t>(t->id())] = 1;
            }
        }
    }
    auto const &all_tiles = graph_.tile_nodes();
    size_t needed_count = 0;
    for (size_t id = 0; id < id_bound; ++id)
    {
        needed_count += needed[id] != 0 ? 1 : 0;
    }
    double const need_ms = cbrk ? breakdown_ms_since(t_need) : 0.0;
    auto const t_loop = std::chrono::steady_clock::now();

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
            !needed[static_cast<std::size_t>(tile_key->id())])
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
        auto *node = const_cast<TileGraph::TileNode *>(tile_key);

        switch (dtype)
        {
        case DataType::FP32:
            allocate_tile_and_register<nntile::fp32_t>(
                node, tile_key->shape());
            break;
        case DataType::FP32_FAST_TF32:
            allocate_tile_and_register<nntile::fp32_fast_tf32_t>(
                node, tile_key->shape());
            break;
        case DataType::FP32_FAST_FP16:
            allocate_tile_and_register<nntile::fp32_fast_fp16_t>(
                node, tile_key->shape());
            break;
        case DataType::FP32_FAST_BF16:
            allocate_tile_and_register<nntile::fp32_fast_bf16_t>(
                node, tile_key->shape());
            break;
        case DataType::FP64:
            allocate_tile_and_register<nntile::fp64_t>(
                node, tile_key->shape());
            break;
        case DataType::FP16:
            allocate_tile_and_register<nntile::fp16_t>(
                node, tile_key->shape());
            break;
        case DataType::BF16:
            allocate_tile_and_register<nntile::bf16_t>(
                node, tile_key->shape());
            break;
        case DataType::INT64:
            allocate_tile_and_register<nntile::int64_t>(
                node, tile_key->shape());
            break;
        case DataType::BOOL:
            allocate_tile_and_register<nntile::bool_t>(
                node, tile_key->shape());
            break;
        default:
            throw std::runtime_error(
                "Unsupported data type for tile allocation");
        }
    };

    // Only touch pending I/O, DCE-live tiles, and newly lowered tile nodes.
    // The loops below are bounded by the pending window's id span
    // (id_bound) or by live_tile_nodes_ (whose ids are edges of surviving
    // pending ops, hence < id_bound) - the historical tail of
    // tile_nodes() above the window's max id is never scanned or
    // zero-initialized, so this stays O(pending phase), not O(session
    // length). New nodes cover ingress staging lowered before any scatter
    // op is appended to execution_order_.
    // Count freshly-allocated tiles only (adoption from tile_adoption_
    // is NOT a fresh allocation). No per-tile clock brackets: the alloc
    // line's loop_ms is the labeled pass total (a nested sub-line was
    // both double-counted inside it and structurally zero in its
    // livechk field - deleted in review).
    std::size_t n_fresh = 0;
    for (size_t id = 0; id < id_bound; ++id)
    {
        // fresh = will actually allocate (payloadless AND not adoptable
        // from tile_adoption_) - adoption is not a fresh allocation.
        if (needed[id] && !all_tiles[id]->has_payload()
            && tile_adoption_.find(all_tiles[id].get())
                == tile_adoption_.end())
        {
            ++n_fresh;
        }
    }
    for (const auto *tile : live_tile_nodes_)
    {
        if (tile != nullptr && !needed[static_cast<size_t>(tile->id())])
        {
            try_allocate(tile, true);
        }
    }
    if (compiled_tile_node_count_ > all_tiles.size())
    {
        compiled_tile_node_count_ = 0;
    }
    for (size_t i = compiled_tile_node_count_; i < all_tiles.size(); ++i)
    {
        try_allocate(all_tiles[i].get(), false);
    }
    compiled_tile_node_count_ = all_tiles.size();
    if (cbrk)
    {
        std::fprintf(stderr,
            "[nntile-cbrk] alloc pending=%zu need_ms=%.3f loop_ms=%.3f "
            "needed=%zu live_slots=%zu all_tiles=%zu fresh=%zu\n",
            n - begin, need_ms, breakdown_ms_since(t_loop), needed_count,
            live_tile_nodes_.size(), all_tiles.size(), n_fresh);
    }
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
#ifdef NNTILE_USE_NNHAUL
        if (payload)
        {
            nnhaul_retained_payloads_.push_back(payload);
        }
#endif
        auto *mut = const_cast<TileGraph::TileNode *>(tile);
        mut->clear_payload();
        sync_logical_starpu_flag(mut);
    }
    queued_dead_tiles_.clear();
}

void Runtime::eliminate_dead_ops(PendingPhaseSnapshot *snap_in)
{
    live_tile_nodes_.clear();
    size_t const n = execution_order_.size();
    if (n == 0)
    {
        return;
    }
    // Already-executed prefix is immutable for incremental sessions; DCE
    // only the pending suffix so compile stays O(new phase).
    size_t const pending_begin =
        executed_op_end_ < n ? executed_op_end_ : n;
    if (pending_begin >= n)
    {
        return;
    }
    size_t const pending = n - pending_begin;

    bool const cbrk = compile_breakdown_enabled();
    auto const t_fill = std::chrono::steady_clock::now();

    // Shared per-compile snapshot (compile() builds it once); fall back to
    // a local build for direct calls.
    PendingPhaseSnapshot own_snap;
    if (snap_in == nullptr || snap_in->op_begin != pending_begin
        || snap_in->in_off.size() != pending + 1)
    {
        own_snap = build_pending_phase_snapshot();
        snap_in = &own_snap;
    }
    PendingPhaseSnapshot const &snap = *snap_in;

    // CSR adjacency over dense tile ids: TileNode::id() is the absolute
    // slot index in TileGraph::tile_nodes() (session-monotonic, GC holes
    // included), so producer/consumer adjacency lives in contiguous
    // offset arrays plus flat edge arrays of pending-local op indices.
    // Indices stay ABSOLUTE; only the extents shrink to the pending
    // window: the arrays are sized to the max tile id over the
    // snapshot's op edges + 1 (every edge tile id is below that bound by
    // construction), so allocate/zero/scan cost scales with the pending
    // phase, not with the session-wide tile_nodes() high-water. Two-pass
    // count-then-fill. This replaces the old hash-map-of-vectors
    // adjacency (~2 x pending map nodes with per-element heap vectors)
    // whose scattered layout made the per-element cost grow with graph
    // size (capacity superlinearity).
    size_t max_pending_id = 0;
    for (TileNode const *t : snap.in_edges)
    {
        if (t != nullptr)
        {
            size_t const id = static_cast<size_t>(t->id());
            if (id > max_pending_id)
            {
                max_pending_id = id;
            }
        }
    }
    for (TileNode const *t : snap.out_edges)
    {
        if (t != nullptr)
        {
            size_t const id = static_cast<size_t>(t->id());
            if (id > max_pending_id)
            {
                max_pending_id = id;
            }
        }
    }
    size_t const id_bound = max_pending_id + 1;
    std::vector<std::uint32_t> prod_off(id_bound + 1, 0);
    std::vector<std::uint32_t> cons_off(id_bound + 1, 0);
    for (size_t local = 0; local < pending; ++local)
    {
        for (std::uint32_t e = snap.out_off[local];
             e < snap.out_off[local + 1]; ++e)
        {
            TileNode const *t = snap.out_edges[e];
            if (t != nullptr)
            {
                ++prod_off[static_cast<size_t>(t->id()) + 1];
            }
        }
        for (std::uint32_t e = snap.in_off[local];
             e < snap.in_off[local + 1]; ++e)
        {
            TileNode const *t = snap.in_edges[e];
            if (t != nullptr)
            {
                ++cons_off[static_cast<size_t>(t->id()) + 1];
            }
        }
    }
    for (size_t id = 0; id < id_bound; ++id)
    {
        prod_off[id + 1] += prod_off[id];
        cons_off[id + 1] += cons_off[id];
    }
    std::vector<std::uint32_t> prod_edges(prod_off[id_bound]);
    std::vector<std::uint32_t> cons_edges(cons_off[id_bound]);
    {
        std::vector<std::uint32_t> prod_cursor(prod_off.begin(),
            prod_off.end() - 1);
        std::vector<std::uint32_t> cons_cursor(cons_off.begin(),
            cons_off.end() - 1);
        for (size_t local = 0; local < pending; ++local)
        {
            for (std::uint32_t e = snap.out_off[local];
                 e < snap.out_off[local + 1]; ++e)
            {
                TileNode const *t = snap.out_edges[e];
                if (t != nullptr)
                {
                    prod_edges[prod_cursor[static_cast<size_t>(t->id())]++] =
                        static_cast<std::uint32_t>(local);
                }
            }
            for (std::uint32_t e = snap.in_off[local];
                 e < snap.in_off[local + 1]; ++e)
            {
                TileNode const *t = snap.in_edges[e];
                if (t != nullptr)
                {
                    cons_edges[cons_cursor[static_cast<size_t>(t->id())]++] =
                        static_cast<std::uint32_t>(local);
                }
            }
        }
    }
    double const fill_ms = cbrk ? breakdown_ms_since(t_fill) : 0.0;
    auto const t_seed = std::chrono::steady_clock::now();

    // Live-tile bitmap + work list of dense tile ids (absolute ids,
    // bounded by the pending window's max id as above). The mark closure
    // is the same reachability fixpoint as the old hash-set version; set
    // contents are order-independent, only scratch layout changed.
    std::vector<std::uint8_t> live_tile(id_bound, 0);
    std::vector<std::uint32_t> work;
    work.reserve(1024);
    auto seed_tile = [&](TileNode const *tile)
    {
        if (tile != nullptr)
        {
            size_t const id = static_cast<size_t>(tile->id());
            if (!live_tile[id])
            {
                live_tile[id] = 1;
                work.push_back(static_cast<std::uint32_t>(id));
            }
        }
    };

    // Seed from tiles whose logical still has a live TensorRef.
    // TILE_UNREGISTER is recorded when that TensorRef is already gone;
    // the StarPU handle stays until this op runs, so seed its inputs
    // even though tile_logical_is_live is false.
    for (size_t local = 0; local < pending; ++local)
    {
        bool const unregister = snap.is_unregister[local] != 0;
        for (std::uint32_t e = snap.out_off[local];
             e < snap.out_off[local + 1]; ++e)
        {
            TileNode const *t = snap.out_edges[e];
            if (t != nullptr && tile_logical_is_live(t))
            {
                seed_tile(t);
            }
        }
        for (std::uint32_t e = snap.in_off[local];
             e < snap.in_off[local + 1]; ++e)
        {
            TileNode const *t = snap.in_edges[e];
            if (t != nullptr
                && (unregister || tile_logical_is_live(t)))
            {
                seed_tile(t);
            }
        }
    }

    // Keep wire-observable sinks: tiles produced but never consumed in
    // the pending window have no daemon-side logical refs - their only
    // reader is the remote gather. This fallback used to run only when
    // the work list was empty, so a single pending TILE_UNREGISTER
    // disabled it and DCE pruned the producing op; the gathered tile
    // stayed uninitialized and StarPU aborted on
    // _starpu_data_check_initialized. Seed sinks unconditionally -
    // liveness of other tiles is unaffected (their consumers still
    // propagate through the work list). Ids above the window's max have
    // no producers in the window, so scanning to id_bound sees every
    // sink there is.
    for (size_t id = 0; id < id_bound; ++id)
    {
        if (prod_off[id + 1] > prod_off[id]
            && cons_off[id + 1] == cons_off[id] && !live_tile[id])
        {
            live_tile[id] = 1;
            work.push_back(static_cast<std::uint32_t>(id));
        }
    }
    if (work.empty())
    {
        if (cbrk)
        {
            std::fprintf(stderr,
                "[nntile-cbrk] dce pending_pre=%zu fill_ms=%.3f seed_ms=%.3f "
                "mark_ms=0.000 compact_ms=0.000 live_ops=0 live_tiles=0\n",
                pending, fill_ms, breakdown_ms_since(t_seed));
        }
        return;
    }
    double const seed_ms = cbrk ? breakdown_ms_since(t_seed) : 0.0;
    auto const t_mark = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> live_op(pending, 0);
    size_t live_op_count = 0;
    auto mark_local = [&](size_t local)
    {
        if (live_op[local] != 0)
        {
            return;
        }
        live_op[local] = 1;
        ++live_op_count;
        // enqueue_op_io over the snapshot: newly marked ops seed their
        // I/O tiles into the work list (dedup via the live bitmap).
        for (std::uint32_t e = snap.in_off[local];
             e < snap.in_off[local + 1]; ++e)
        {
            seed_tile(snap.in_edges[e]);
        }
        for (std::uint32_t e = snap.out_off[local];
             e < snap.out_off[local + 1]; ++e)
        {
            seed_tile(snap.out_edges[e]);
        }
    };

    size_t wi = 0;
    while (wi < work.size())
    {
        size_t const id = work[wi++];
        for (std::uint32_t k = prod_off[id]; k < prod_off[id + 1]; ++k)
        {
            mark_local(prod_edges[k]);
        }
        // Consumers of a live tile may be sinks (empty outputs).
        for (std::uint32_t k = cons_off[id]; k < cons_off[id + 1]; ++k)
        {
            mark_local(cons_edges[k]);
        }
    }
    double const mark_ms = cbrk ? breakdown_ms_since(t_mark) : 0.0;
    auto const t_compact = std::chrono::steady_clock::now();

    // Keep the executed prefix in place. Rebuilding the full vector every
    // compile recopied O(history) shared_ptrs and made step time grow.
    if (live_op_count == pending)
    {
        if (cbrk)
        {
            std::fprintf(stderr,
                "[nntile-cbrk] dce pending_pre=%zu fill_ms=%.3f seed_ms=%.3f "
                "mark_ms=%.3f compact_ms=0.000 live_ops=%zu "
                "live_tiles=%zu\n",
                pending, fill_ms, seed_ms, mark_ms, live_op_count,
                work.size());
        }
        live_tile_nodes_.reserve(work.size());
        for (std::uint32_t id : work)
        {
            live_tile_nodes_.insert(graph_.tile_nodes()[id].get());
        }
        return;
    }
    // No survivor bits are published: allocate_missing_tiles detects the
    // shrunken execution_order_ (snapshot op_begin/pending mismatch) and
    // rebuilds from the post-DCE order, which is what guarantees its
    // needed-walk sees exactly the surviving ops.
    size_t write = pending_begin;
    for (size_t i = pending_begin; i < n; ++i)
    {
        if (live_op[i - pending_begin] == 0)
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
    if (cbrk)
    {
        std::fprintf(stderr,
            "[nntile-cbrk] dce pending_pre=%zu fill_ms=%.3f seed_ms=%.3f "
            "mark_ms=%.3f compact_ms=%.3f live_ops=%zu live_tiles=%zu\n",
            pending, fill_ms, seed_ms, mark_ms,
            breakdown_ms_since(t_compact), live_op_count, work.size());
    }
    live_tile_nodes_.reserve(work.size());
    for (std::uint32_t id : work)
    {
        live_tile_nodes_.insert(graph_.tile_nodes()[id].get());
    }
}

std::string Runtime::execution_op_name(size_t i) const
{
    if (i >= execution_order_.size())
    {
        throw std::out_of_range(
            "Runtime::execution_op_name: index");
    }
    return execution_order_[i]->op_name();
}

void Runtime::wait()
{
    starpu_task_wait_for_all_counted();
#ifdef NNTILE_USE_NNHAUL
    nnhaul_retained_payloads_.clear();
#endif
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
