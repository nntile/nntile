/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.cpp
 */

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"

#include "nntile_context.h"

#include <ATen/Tensor.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/runtime.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>
#include <nntile/tensor/ops/scatter.hh>
#include <nntile/tensor/tensor_graph_tiling.hh>
#include <nntile/tile/graph.hh>

#include <starpu.h>

namespace nntile
{
void apply_tiling_to_axis(
    AxisDescriptor *ad,
    const std::vector<Index> &sizes);
std::vector<Index> tile_sizes_for_axis_extent(
    const std::vector<Index> &pattern,
    Index extent);
} // namespace nntile

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace torch_nntile
{

namespace
{

struct MappedTensor
{
    nntile::TensorGraph::TensorNode *node = nullptr;
    nntile::TensorGraph::TensorNode *staging_node = nullptr;
    nntile::DataType dtype = nntile::DataType::FP32;
    std::size_t count = 0;
    bool needs_host_copy = false;
    bool bind_at_execute = false;
    bool is_persistent_input = false;
};

nntile::TensorGraph::TensorNode *bind_target_node(const MappedTensor &mapped)
{
    if (mapped.staging_node != nullptr)
    {
        return mapped.staging_node;
    }
    return mapped.node;
}

std::mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
std::unordered_map<void *, MappedTensor> g_tensor_nodes;
std::unordered_set<nntile::TensorGraph::TensorNode *> g_all_nodes;
std::vector<at::Tensor> g_pinned_tensors;
std::unordered_map<void *, std::unordered_map<int, std::string>> g_axis_name_hints;
std::unordered_map<std::string, std::vector<nntile::Index>> g_axis_tiling_by_name;

struct GraphSession
{
    std::unique_ptr<nntile::TensorGraph> tensor_graph;
    std::unique_ptr<nntile::TileGraph> tile_graph;
    std::unique_ptr<nntile::Runtime> runtime;
    std::unordered_map<void *, nntile::TensorGraph::TensorNode *> storage_to_node;
};

std::unique_ptr<GraphSession> g_session;
std::unordered_map<void *, std::vector<std::shared_ptr<void>>> g_persisted_tiles_by_storage;
//! Keeps every session tile buffer alive until recorder shutdown.
std::vector<std::shared_ptr<void>> g_persisted_tile_pool;

void log_tile_adoption(const std::string &message)
{
    if (is_context_verbose())
    {
        std::cerr << "[torch_nntile tile_adoption] " << message << '\n';
    }
}

const char *tensor_node_label(nntile::TensorGraph::TensorNode const *node)
{
    if (node == nullptr)
    {
        return "<null>";
    }
    return node->name().empty() ? "<unnamed>" : node->name().c_str();
}

void assign_axis_group_name(nntile::AxisDescriptor *axis, const std::string &name)
{
    if (axis == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile set_axis_group_name: invalid axis");
    }
    if (!axis->name.empty() && axis->name != name)
    {
        throw std::runtime_error(
            "torch_nntile set_axis_group_name: axis group already named '" +
            axis->name + "', cannot rename to '" + name + "'");
    }
    axis->name = name;
}

void apply_axis_name_hints_locked(void *data_ptr, nntile::TensorGraph::TensorNode *node)
{
    const auto hints = g_axis_name_hints.find(data_ptr);
    if (hints == g_axis_name_hints.end())
    {
        return;
    }
    for (const auto &[dim, name] : hints->second)
    {
        if (dim < 0 || dim >= node->ndim())
        {
            throw std::runtime_error(
                "torch_nntile set_axis_group_name: dimension out of range");
        }
        assign_axis_group_name(node->axis(dim), name);
    }
}

void apply_pending_axis_tiling_locked()
{
    if (g_graph == nullptr || g_axis_tiling_by_name.empty())
    {
        return;
    }

    for (const auto &[name, pattern] : g_axis_tiling_by_name)
    {
        bool found_any = false;
        for (nntile::AxisDescriptor *axis : g_graph->axis_groups())
        {
            if (axis == nullptr || axis->name != name)
            {
                continue;
            }
            found_any = true;
            const std::vector<nntile::Index> resolved =
                nntile::tile_sizes_for_axis_extent(pattern, axis->extent);
            nntile::apply_tiling_to_axis(axis, resolved);
        }
        if (!found_any)
        {
            throw std::runtime_error(
                "torch_nntile set_axis_group_tiling: unknown axis group '" +
                name + "'");
        }
    }
}

void track_node(nntile::TensorGraph::TensorNode *node)
{
    if (node != nullptr)
    {
        g_all_nodes.insert(node);
    }
}

nntile::Index graph_numel(const std::vector<nntile::Index> &graph_shape)
{
    nntile::Index nelems = 1;
    for (const nntile::Index dim : graph_shape)
    {
        nelems *= dim;
    }
    return nelems;
}

void copy_tensor_from_runtime(
    nntile::Runtime &runtime,
    const MappedTensor &mapped,
    void *data_ptr)
{
    const std::size_t count = mapped.count;
    if (count == 0 || mapped.node == nullptr)
    {
        return;
    }
    if (!mapped.needs_host_copy && !runtime.is_initialized(mapped.node))
    {
        return;
    }
    switch (mapped.dtype)
    {
    case nntile::DataType::FP32:
    {
        const std::vector<float> result =
            runtime.get_output<float>(mapped.node);
        if (result.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile execute: output size mismatch");
        }
        if (result.data() != data_ptr)
        {
            std::memcpy(data_ptr, result.data(), count * sizeof(float));
        }
        break;
    }
    case nntile::DataType::INT64:
    {
        const std::vector<std::int64_t> result =
            runtime.get_output<std::int64_t>(mapped.node);
        if (result.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile execute: output size mismatch");
        }
        if (result.data() != data_ptr)
        {
            std::memcpy(
                data_ptr,
                result.data(),
                count * sizeof(std::int64_t));
        }
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile graph recorder: unsupported output dtype");
    }
}

void copy_output_if_needed(
    nntile::Runtime &runtime,
    const MappedTensor &mapped,
    void *data_ptr)
{
    if (!mapped.node->is_output())
    {
        return;
    }
    copy_tensor_from_runtime(runtime, mapped, data_ptr);
}

void copy_host_visible_outputs(
    nntile::Runtime &runtime,
    const void * /*preferred_ptr*/)
{
    for (auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        if (!mapped.needs_host_copy)
        {
            continue;
        }
        copy_output_if_needed(runtime, mapped, data_ptr);
    }
}

void bind_storage_to_runtime(
    nntile::Runtime &runtime,
    void *data_ptr,
    const MappedTensor &mapped)
{
    nntile::TensorGraph::TensorNode *target = bind_target_node(mapped);
    if (target == nullptr)
    {
        return;
    }
    const std::size_t count = mapped.count;
    switch (mapped.dtype)
    {
    case nntile::DataType::FP32:
        runtime.bind_data(
            target,
            static_cast<const float *>(data_ptr),
            count);
        break;
    case nntile::DataType::INT64:
        runtime.bind_data(
            target,
            static_cast<const std::int64_t *>(data_ptr),
            count);
        break;
    default:
        throw std::runtime_error(
            "torch_nntile graph recorder: unsupported bind dtype");
    }
}

void capture_persisted_tiles_from_session()
{
    g_persisted_tiles_by_storage.clear();
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        log_tile_adoption("capture: no previous session");
        return;
    }
    std::unordered_map<nntile::TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> all_tiles;
    std::unordered_map<nntile::TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> initialized_tiles;
    g_session->runtime->export_all_tiles(all_tiles);
    g_session->runtime->export_initialized_tiles(initialized_tiles);
    const std::size_t pool_before = g_persisted_tile_pool.size();
    std::size_t tiles_retained = 0;
    for (const auto &[tensor, tiles] : all_tiles)
    {
        (void) tensor;
        for (const auto &tile_ptr : tiles)
        {
            if (tile_ptr != nullptr)
            {
                g_persisted_tile_pool.push_back(tile_ptr);
                ++tiles_retained;
            }
        }
    }
    std::size_t storages_mapped = 0;
    for (const auto &[data_ptr, node] : g_session->storage_to_node)
    {
        if (node == nullptr)
        {
            continue;
        }
        const auto found = initialized_tiles.find(node);
        if (found != initialized_tiles.end())
        {
            g_persisted_tiles_by_storage[data_ptr] = found->second;
            ++storages_mapped;
            if (is_context_verbose())
            {
                log_tile_adoption(
                    "capture: storage=" + std::to_string(
                        reinterpret_cast<std::uintptr_t>(data_ptr)) +
                    " tensor='" + std::string(tensor_node_label(node)) +
                    "' tiles=" + std::to_string(found->second.size()));
            }
        }
    }
    if (is_context_verbose())
    {
        log_tile_adoption(
            "capture: all_tensors=" + std::to_string(all_tiles.size()) +
            " initialized_tensors=" + std::to_string(initialized_tiles.size()) +
            " storages_for_adoption=" + std::to_string(storages_mapped) +
            " tiles_retained=" + std::to_string(tiles_retained) +
            " pool_size=" + std::to_string(pool_before) + "->" +
            std::to_string(g_persisted_tile_pool.size()));
    }
}

void stage_persisted_tiles_for_session(
    nntile::Runtime &runtime,
    const nntile::TileGraph &tile_graph)
{
    if (g_persisted_tiles_by_storage.empty())
    {
        log_tile_adoption("stage: no persisted storages");
        return;
    }
    nntile::TensorNodeToTileMap tile_map;
    for (const auto &uptr : tile_graph.tensor_descriptors())
    {
        const nntile::TileGraph::TensorDescriptor &desc = *uptr;
        if (desc.source_node != nullptr)
        {
            tile_map[desc.source_node] = desc.tiles;
        }
    }
    std::unordered_map<nntile::TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> by_node;
    for (const auto &[data_ptr, tiles] : g_persisted_tiles_by_storage)
    {
        const auto mapped = g_tensor_nodes.find(data_ptr);
        if (mapped == g_tensor_nodes.end())
        {
            log_tile_adoption(
                "stage: skip storage=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(data_ptr)) +
                " (no pending tensor node)");
            continue;
        }
        nntile::TensorGraph::TensorNode const *node =
            bind_target_node(mapped->second);
        if (node == nullptr)
        {
            log_tile_adoption(
                "stage: skip storage=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(data_ptr)) +
                " (no pending tensor node)");
            continue;
        }
        const auto tm_it = tile_map.find(node);
        if (tm_it == tile_map.end())
        {
            log_tile_adoption(
                "stage: skip storage=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(data_ptr)) +
                " tensor='" + std::string(tensor_node_label(node)) +
                "' (not in new tile_map)");
            continue;
        }
        if (tm_it->second.size() != tiles.size())
        {
            log_tile_adoption(
                "stage: skip storage=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(data_ptr)) +
                " tensor='" + std::string(tensor_node_label(node)) +
                "' (tile count mismatch saved=" +
                std::to_string(tiles.size()) + " new=" +
                std::to_string(tm_it->second.size()) + ")");
            continue;
        }
        log_tile_adoption(
            "stage: candidate storage=" + std::to_string(
                reinterpret_cast<std::uintptr_t>(data_ptr)) +
            " tensor='" + std::string(tensor_node_label(node)) +
            "' tiles=" + std::to_string(tiles.size()));
        by_node[node] = tiles;
    }
    const std::vector<nntile::TensorGraph::TensorNode const *> adopted =
        runtime.stage_persisted_tiles(by_node, tile_map);
    for (nntile::TensorGraph::TensorNode const *node : adopted)
    {
        if (node != nullptr)
        {
            log_tile_adoption(
                "stage: adopted tensor='" +
                std::string(tensor_node_label(node)) + "'");
        }
    }
    if (is_context_verbose())
    {
        log_tile_adoption(
            "stage: adopted " + std::to_string(adopted.size()) + " / " +
            std::to_string(by_node.size()) + " candidates");
    }
    std::unordered_map<nntile::TensorGraph::TensorNode const *, bool> init;
    for (nntile::TensorGraph::TensorNode const *node : adopted)
    {
        if (node != nullptr)
        {
            init[node] = true;
        }
    }
    runtime.restore_persisted_init_state(init);
}

void clear_pending_graph_after_compile_locked()
{
    g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    g_all_nodes.clear();
    for (auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        (void) data_ptr;
        mapped.node = nullptr;
        mapped.staging_node = nullptr;
        if (mapped.is_persistent_input)
        {
            mapped.bind_at_execute = true;
        }
    }
    g_pinned_tensors.clear();
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
}

void drain_starpu_after_session_teardown()
{
    if (!starpu_is_initialized())
    {
        return;
    }
    starpu_task_wait_for_all();
    // Session tiles are retained in g_persisted_tile_pool so StarPU handles
    // are not async-unregistered during recompile; wait again after teardown.
    starpu_task_wait_for_all();
}

void insert_input_scatter_staging_locked()
{
    if (g_graph == nullptr)
    {
        return;
    }
    std::vector<std::shared_ptr<nntile::TensorGraph::OpNode>> scatter_ops;
    for (auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        if (mapped.node == nullptr)
        {
            continue;
        }
        if (!mapped.bind_at_execute && !mapped.is_persistent_input)
        {
            continue;
        }
        mapped.staging_node = nullptr;
        const nntile::TensorAxisLayout layout(mapped.node);
        if (layout.grid_volume() <= 1)
        {
            continue;
        }
        auto *staging = g_graph->data(
            mapped.node->shape(),
            mapped.node->dtype());
        staging->mark_input(true);
        staging->set_name(
            std::string("host_") +
            std::to_string(reinterpret_cast<std::uintptr_t>(data_ptr)));
        track_node(staging);
        scatter_ops.push_back(
            std::make_shared<nntile::tensor::TensorScatterOp>(
                staging,
                mapped.node));
        mapped.staging_node = staging;
    }
    if (!scatter_ops.empty())
    {
        g_graph->prepend_ops(std::move(scatter_ops));
    }
}

void compile_graph_locked(bool clear_pending_after = true)
{
    if (g_graph == nullptr || g_graph->num_ops() == 0)
    {
        return;
    }

    ensure_nntile_context();

    for (nntile::TensorGraph::TensorNode *node : g_all_nodes)
    {
        node->mark_output(true);
    }

    apply_pending_axis_tiling_locked();
    insert_input_scatter_staging_locked();
    capture_persisted_tiles_from_session();

    auto compiled_tensor_graph = std::move(g_graph);
    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(*compiled_tensor_graph);
    g_session.reset();
    drain_starpu_after_session_teardown();
    g_session = std::make_unique<GraphSession>();
    g_session->tensor_graph = std::move(compiled_tensor_graph);
    g_session->tile_graph =
        std::make_unique<nntile::TileGraph>(std::move(tile_graph));
    g_session->runtime =
        std::make_unique<nntile::Runtime>(*g_session->tile_graph);
    stage_persisted_tiles_for_session(*g_session->runtime, *g_session->tile_graph);
    g_session->runtime->compile();

    for (const auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        if (mapped.node == nullptr)
        {
            continue;
        }
        if (!mapped.bind_at_execute && !mapped.is_persistent_input)
        {
            continue;
        }
        nntile::TensorGraph::TensorNode *bind_node =
            bind_target_node(mapped);
        if (bind_node == nullptr)
        {
            continue;
        }
        bind_node->mark_input(true);
        if (g_session->runtime->is_initialized(bind_node))
        {
            log_tile_adoption(
                "compile: skip bind (already initialized / adopted) storage=" +
                std::to_string(reinterpret_cast<std::uintptr_t>(data_ptr)) +
                " tensor='" + std::string(tensor_node_label(bind_node)) +
                "'");
            continue;
        }
        log_tile_adoption(
            "compile: bind staging storage=" +
            std::to_string(reinterpret_cast<std::uintptr_t>(data_ptr)) +
            " tensor='" + std::string(tensor_node_label(bind_node)) +
            "' nelems=" + std::to_string(mapped.count));
        bind_storage_to_runtime(*g_session->runtime, data_ptr, mapped);
    }

    for (const auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        nntile::TensorGraph::TensorNode *stored = bind_target_node(mapped);
        if (stored != nullptr)
        {
            g_session->storage_to_node[data_ptr] = stored;
        }
    }

    if (clear_pending_after)
    {
        clear_pending_graph_after_compile_locked();
    }
}

void run_graph_locked()
{
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    g_session->runtime->execute();
    g_session->runtime->wait();
}

void reset_recorder_locked()
{
    g_graph.reset();
    g_tensor_nodes.clear();
    g_all_nodes.clear();
    g_pinned_tensors.clear();
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
    g_session.reset();
    g_persisted_tiles_by_storage.clear();
    g_persisted_tile_pool.clear();
    drain_starpu_after_session_teardown();
}

void execute_pending_graph_locked()
{
    compile_graph_locked(false);
    run_graph_locked();
    if (g_session != nullptr && g_session->runtime != nullptr)
    {
        copy_host_visible_outputs(*g_session->runtime, nullptr);
    }
    clear_pending_graph_after_compile_locked();
    reset_recorder_locked();
}

void shutdown_recorder_locked()
{
    if (g_graph != nullptr && g_graph->num_ops() > 0)
    {
        compile_graph_locked(false);
        run_graph_locked();
        if (g_session != nullptr && g_session->runtime != nullptr)
        {
            copy_host_visible_outputs(*g_session->runtime, nullptr);
        }
    }
    reset_recorder_locked();
}

} // namespace

bool has_pending_graph()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    return g_graph != nullptr && g_graph->num_ops() > 0;
}

void require_no_pending_graph(const char *op_name)
{
    if (!is_graph_mode())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    if (g_graph != nullptr && g_graph->num_ops() > 0)
    {
        throw std::runtime_error(
            std::string("torch_nntile: pending graph must be flushed with "
                        "torch_nntile.execute() before ") +
            op_name);
    }
}

void execute_pending_graph()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    execute_pending_graph_locked();
}

void compile_graph()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    compile_graph_locked(true);
}

void run_graph()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    run_graph_locked();
}

void reset_graph_session()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    reset_recorder_locked();
}

void shutdown_recorder()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    shutdown_recorder_locked();
}

bool has_graph_session()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    return g_session != nullptr && g_session->runtime != nullptr;
}

void sync_nntile_storage_to_runtime(void *data_ptr)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *node = nullptr;
    const auto session_it = g_session->storage_to_node.find(data_ptr);
    if (session_it != g_session->storage_to_node.end())
    {
        node = session_it->second;
    }
    else
    {
        const auto found = g_tensor_nodes.find(data_ptr);
        if (found != g_tensor_nodes.end())
        {
            node = found->second.node;
        }
    }
    if (node == nullptr)
    {
        return;
    }
    MappedTensor mapped;
    const auto found = g_tensor_nodes.find(data_ptr);
    if (found != g_tensor_nodes.end())
    {
        mapped = found->second;
        mapped.node = node;
    }
    else
    {
        mapped.node = node;
        mapped.dtype = node->dtype();
        mapped.count = static_cast<std::size_t>(node->nelems());
    }
    bind_storage_to_runtime(*g_session->runtime, data_ptr, mapped);
}

void sync_runtime_to_nntile_storage(void *data_ptr)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *node = nullptr;
    const auto session_it = g_session->storage_to_node.find(data_ptr);
    if (session_it != g_session->storage_to_node.end())
    {
        node = session_it->second;
    }
    else
    {
        const auto found = g_tensor_nodes.find(data_ptr);
        if (found != g_tensor_nodes.end())
        {
            node = found->second.node;
        }
    }
    if (node == nullptr)
    {
        return;
    }
    g_session->runtime->wait();
    MappedTensor mapped;
    mapped.node = node;
    mapped.dtype = node->dtype();
    mapped.count = static_cast<std::size_t>(node->nelems());
    const auto found = g_tensor_nodes.find(data_ptr);
    if (found != g_tensor_nodes.end())
    {
        mapped.needs_host_copy = found->second.needs_host_copy;
    }
    else
    {
        mapped.needs_host_copy = true;
    }
    copy_tensor_from_runtime(*g_session->runtime, mapped, data_ptr);
}

void maybe_execute_after_record()
{
    if (get_runtime_mode() == RuntimeMode::Eager)
    {
        execute_pending_graph();
    }
}

nntile::TensorGraph &recorder_graph()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    }
    return *g_graph;
}

nntile::TensorGraph::TensorNode *get_or_create_data_node(
    void *data_ptr,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    }

    const auto found = g_tensor_nodes.find(data_ptr);
    if (found != g_tensor_nodes.end() && found->second.node != nullptr)
    {
        // Intermediate op output fed into a later op: keep the node for DCE
        // but stop binding/copying through storage PyTorch may free or reuse.
        if (!found->second.is_persistent_input && found->second.needs_host_copy)
        {
            found->second.bind_at_execute = false;
            found->second.needs_host_copy = false;
        }
        track_node(found->second.node);
        return found->second.node;
    }

    const bool is_persistent =
        found != g_tensor_nodes.end() && found->second.is_persistent_input;
    const bool bind_at_execute =
        (found != g_tensor_nodes.end() && found->second.bind_at_execute) ||
        mark_as_input;

    auto *node = g_graph->data(shape, dtype);
    if (mark_as_input || is_persistent)
    {
        node->mark_input(true);
    }
    apply_axis_name_hints_locked(data_ptr, node);
    track_node(node);
    g_tensor_nodes[data_ptr] = MappedTensor{
        node,
        nullptr,
        dtype,
        static_cast<std::size_t>(graph_numel(shape)),
        false,
        bind_at_execute,
        is_persistent || mark_as_input};
    return node;
}

void register_data_node(
    void *data_ptr,
    nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    node->mark_output(true);
    track_node(node);

    const auto found = g_tensor_nodes.find(data_ptr);
    if (found != g_tensor_nodes.end() && found->second.is_persistent_input)
    {
        // In-place update of optimizer state / weights: keep execute-time bind
        // and gather updated values back to host after execute / .cpu().
        found->second.node = node;
        found->second.dtype = node->dtype();
        found->second.count = static_cast<std::size_t>(node->nelems());
        found->second.needs_host_copy = true;
        return;
    }

    g_tensor_nodes[data_ptr] = MappedTensor{
        node,
        nullptr,
        node->dtype(),
        static_cast<std::size_t>(node->nelems()),
        true,
        false,
        false};
}

nntile::TensorGraph::TensorNode *lookup_data_node(void *data_ptr)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    const auto found = g_tensor_nodes.find(data_ptr);
    if (found == g_tensor_nodes.end())
    {
        return nullptr;
    }
    return found->second.node;
}

void track_graph_node(nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    track_node(node);
}

void pin_tensor_for_graph(const at::Tensor &tensor)
{
    if (!is_graph_mode())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    g_pinned_tensors.push_back(tensor);
}

void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs)
{
    if (!is_graph_mode())
    {
        return;
    }
    for (const at::Tensor &tensor : inputs)
    {
        pin_tensor_for_graph(tensor);
    }
}

void pin_graph_op_output(const at::Tensor &output, bool pin_output)
{
    if (!is_graph_mode() || !pin_output)
    {
        return;
    }
    pin_tensor_for_graph(output);
}

void set_axis_group_name(
    void *data_ptr,
    int ndim,
    const std::unordered_map<int, std::string> &names)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    for (const auto &[dim, name] : names)
    {
        if (dim < 0 || dim >= ndim)
        {
            throw std::runtime_error(
                "torch_nntile set_axis_group_name: dimension out of range");
        }
        if (name.empty())
        {
            throw std::runtime_error(
                "torch_nntile set_axis_group_name: name must be non-empty");
        }

        nntile::TensorGraph::TensorNode *node = nullptr;
        const auto found = g_tensor_nodes.find(data_ptr);
        if (found != g_tensor_nodes.end())
        {
            node = found->second.node;
        }
        if (node != nullptr)
        {
            assign_axis_group_name(node->axis(dim), name);
        }
        else
        {
            g_axis_name_hints[data_ptr][dim] = name;
        }
    }
}

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes)
{
    if (name.empty())
    {
        throw std::runtime_error(
            "torch_nntile set_axis_group_tiling: name must be non-empty");
    }
    if (tile_sizes.empty())
    {
        throw std::runtime_error(
            "torch_nntile set_axis_group_tiling: tile_sizes must be non-empty");
    }
    std::vector<nntile::Index> pattern(tile_sizes.begin(), tile_sizes.end());
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    g_axis_tiling_by_name[name] = std::move(pattern);
}

std::string format_pending_tile_sizes(
    const std::vector<nntile::Index> &sizes)
{
    if (sizes.empty())
    {
        return "";
    }
    if (sizes.size() == 1)
    {
        return std::to_string(sizes.front());
    }
    std::ostringstream ss;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        if (i > 0)
        {
            ss << ',';
        }
        ss << sizes[i];
    }
    return ss.str();
}

std::string format_axis_groups_locked()
{
    if (g_graph == nullptr)
    {
        return "Axis groups: (no pending graph)\n";
    }

    const std::vector<nntile::AxisDescriptor *> groups = g_graph->axis_groups();
    std::size_t tiled = 0;
    for (const nntile::AxisDescriptor *group : groups)
    {
        if (group != nullptr && group->is_tiled())
        {
            ++tiled;
        }
    }

    std::ostringstream ss;
    ss << "Pending TensorGraph: data=" << g_graph->num_data()
       << ", ops=" << g_graph->num_ops() << ", axis_groups=" << groups.size()
       << ", tiled=" << tiled << '/' << groups.size() << '\n';
    if (groups.empty())
    {
        return ss.str();
    }

    ss << "Axis groups:\n";
    for (const nntile::AxisDescriptor *group : groups)
    {
        if (group == nullptr)
        {
            continue;
        }
        ss << "  extent=" << group->extent;
        if (!group->name.empty())
        {
            ss << " name='" << group->name << '\'';
        }
        if (group->is_tiled())
        {
            ss << " tile=" << group->tile_sizes_to_string();
        }
        else if (!group->name.empty())
        {
            const auto pending = g_axis_tiling_by_name.find(group->name);
            if (pending != g_axis_tiling_by_name.end())
            {
                ss << " pending_tile=" << format_pending_tile_sizes(pending->second);
            }
        }
        ss << " members=" << group->members.size() << '\n';
    }
    return ss.str();
}

std::string format_axis_groups()
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    return format_axis_groups_locked();
}

void print_axis_groups()
{
    const std::string text = format_axis_groups();
    if (!text.empty())
    {
        std::fputs(text.c_str(), stdout);
        if (text.back() != '\n')
        {
            std::fputc('\n', stdout);
        }
        std::fflush(stdout);
    }
}

} // namespace torch_nntile

#else

namespace torch_nntile
{

namespace
{

[[noreturn]] void require_libnntile()
{
    throw std::runtime_error(
        "torch_nntile graph recorder requires libnntile "
        "(rebuild with NNTILE_BUILD_DIR set)");
}

} // namespace

bool has_pending_graph()
{
    return false;
}

void require_no_pending_graph(const char * /*op_name*/)
{
}

void execute_pending_graph()
{
    require_libnntile();
}

void compile_graph()
{
    require_libnntile();
}

void run_graph()
{
    require_libnntile();
}

void reset_graph_session()
{
    require_libnntile();
}

bool has_graph_session()
{
    return false;
}

void sync_nntile_storage_to_runtime(void * /*data_ptr*/)
{
}

void sync_runtime_to_nntile_storage(void * /*data_ptr*/)
{
}

void maybe_execute_after_record()
{
}

void pin_tensor_for_graph(const at::Tensor & /*tensor*/)
{
}

void pin_graph_op_inputs(const std::vector<at::Tensor> & /*inputs*/)
{
}

void pin_graph_op_output(const at::Tensor & /*output*/, bool /*pin_output*/)
{
}

void set_axis_group_name(
    void * /*data_ptr*/,
    int /*ndim*/,
    const std::unordered_map<int, std::string> & /*names*/)
{
    require_libnntile();
}

void set_axis_group_tiling(
    const std::string & /*name*/,
    const std::vector<std::int64_t> & /*tile_sizes*/)
{
    require_libnntile();
}

std::string format_axis_groups()
{
    require_libnntile();
    return {};
}

void print_axis_groups()
{
    require_libnntile();
}

} // namespace torch_nntile

#endif
