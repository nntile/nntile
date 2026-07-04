/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.cpp
 */

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"

#include "nntile_context.h"

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <stdexcept>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/runtime.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>
#include <nntile/tensor/ops/scatter.hh>
#include <nntile/tensor/ops/contiguous_view.hh>
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
#include <cstdlib>
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
    void *host_data_ptr = nullptr;
};

nntile::TensorGraph::TensorNode *bind_target_node(const MappedTensor &mapped)
{
    if (mapped.staging_node != nullptr)
    {
        return mapped.staging_node;
    }
    return mapped.node;
}

std::recursive_mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
std::unordered_map<TensorImplKey, MappedTensor> g_tensor_nodes;
std::unordered_map<TensorImplKey, nntile::TensorGraph::TensorNode *>
    g_param_grad_nodes;
struct ParamGradEntry
{
    nntile::TensorGraph::TensorNode *grad_node = nullptr;
    at::Tensor param;
};
std::unordered_map<TensorImplKey, ParamGradEntry> g_param_grad_registry;
std::vector<nntile::TensorGraph::TensorNode *> g_relu_preactivation_stack;
std::unordered_set<nntile::TensorGraph::TensorNode *> g_all_nodes;
std::vector<at::Tensor> g_pinned_tensors;
std::unordered_map<TensorImplKey, std::unordered_map<int, std::string>>
    g_axis_name_hints;
std::unordered_map<std::string, std::vector<nntile::Index>> g_axis_tiling_by_name;

struct GraphSession
{
    std::unique_ptr<nntile::TensorGraph> tensor_graph;
    std::unique_ptr<nntile::TileGraph> tile_graph;
    std::unique_ptr<nntile::Runtime> runtime;
    std::unordered_map<TensorImplKey, nntile::TensorGraph::TensorNode *>
        impl_to_node;
};

std::unique_ptr<GraphSession> g_session;
std::unordered_map<TensorImplKey, std::vector<std::shared_ptr<void>>>
    g_persisted_tiles_by_impl;
//! Keeps every session tile buffer alive until recorder shutdown.
std::vector<std::shared_ptr<void>> g_persisted_tile_pool;

void sync_param_grad_aliases_locked();

void register_grad_alias_for_host_copy_locked(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node);

void log_tile_adoption(const std::string &message)
{
    if (is_context_verbose())
    {
        std::cerr << "[torch_nntile tile_adoption] " << message << '\n';
    }
}

//! Tile adoption applies to staged tensors whose runtime tiles are
//! authoritative across recompiles (weights / optimizer state). Grad aliases
//! set needs_host_copy but keep bind_at_execute=false; ephemeral staged
//! inputs (x, labels) keep needs_host_copy=false and rebind from host.
bool should_adopt_tiles_for_mapped(const MappedTensor &mapped)
{
    return mapped.needs_host_copy && mapped.bind_at_execute;
}

bool should_retain_mapped_tensor_after_compile(
    TensorImplKey impl_key,
    const MappedTensor &mapped)
{
    return is_staged_input_impl(impl_key) || mapped.needs_host_copy ||
        mapped.is_persistent_input;
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

bool graph_shape_matches_node(
    const std::vector<nntile::Index> &shape,
    nntile::TensorGraph::TensorNode *node)
{
    if (node == nullptr)
    {
        return false;
    }
    if (static_cast<std::size_t>(node->ndim()) != shape.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (node->shape()[i] != shape[i])
        {
            return false;
        }
    }
    return true;
}

void apply_axis_name_hints_locked(
    TensorImplKey key,
    nntile::TensorGraph::TensorNode *node)
{
    const auto hints = g_axis_name_hints.find(key);
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

bool shapes_equal(
    const std::vector<nntile::Index> &lhs,
    const std::vector<nntile::Index> &rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i] != rhs[i])
        {
            return false;
        }
    }
    return true;
}

nntile::TensorGraph::TensorNode *ensure_view_alias_locked(
    nntile::TensorGraph::TensorNode *src,
    const std::vector<nntile::Index> &view_shape,
    nntile::DataType dtype)
{
    if (shapes_equal(src->shape(), view_shape))
    {
        return src;
    }
    if (graph_numel(src->shape()) != graph_numel(view_shape))
    {
        throw std::invalid_argument(
            "view: storage alias must preserve numel");
    }
    auto *view_node = g_graph->data(view_shape, dtype)->set_name("view");
    track_node(view_node);
    nntile::tensor::contiguous_view(src, view_node);
    return view_node;
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

nntile::TensorGraph::TensorNode *session_node_for_impl_locked(
    TensorImplKey impl_key)
{
    if (g_session != nullptr)
    {
        const auto session_it = g_session->impl_to_node.find(impl_key);
        if (session_it != g_session->impl_to_node.end())
        {
            return session_it->second;
        }
    }
    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end())
    {
        return found->second.node;
    }
    return nullptr;
}

void copy_host_visible_outputs(
    nntile::Runtime &runtime,
    const void * /*preferred_ptr*/)
{
    for (auto &[impl_key, mapped] : g_tensor_nodes)
    {
        (void) impl_key;
        if (!mapped.needs_host_copy || mapped.host_data_ptr == nullptr)
        {
            continue;
        }
        if (mapped.node != nullptr && mapped.node->is_output())
        {
            copy_output_if_needed(runtime, mapped, mapped.host_data_ptr);
        }
        else
        {
            copy_tensor_from_runtime(
                runtime,
                mapped,
                mapped.host_data_ptr);
        }
    }
}

void sync_current_run_visible_outputs_locked()
{
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    sync_param_grad_aliases_locked();
    copy_host_visible_outputs(*g_session->runtime, nullptr);
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
    case nntile::DataType::BOOL:
        runtime.bind_data(
            target,
            reinterpret_cast<const bool *>(data_ptr),
            count);
        break;
    default:
        throw std::runtime_error(
            "torch_nntile graph recorder: unsupported bind dtype");
    }
}

void capture_persisted_tiles_from_session()
{
    g_persisted_tiles_by_impl.clear();
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        log_tile_adoption("capture: no previous session");
        return;
    }
    std::unordered_map<nntile::TensorGraph::TensorNode const *,
        std::vector<std::shared_ptr<void>>> initialized_tiles;
    g_session->runtime->export_initialized_tiles(initialized_tiles);
    const std::size_t pool_before = g_persisted_tile_pool.size();
    std::size_t tiles_retained = 0;
    std::size_t storages_mapped = 0;
    for (const auto &[impl_key, node] : g_session->impl_to_node)
    {
        if (node == nullptr)
        {
            continue;
        }
        const auto mapped = g_tensor_nodes.find(impl_key);
        if (mapped == g_tensor_nodes.end() ||
            !should_adopt_tiles_for_mapped(mapped->second))
        {
            continue;
        }
        const auto found = initialized_tiles.find(node);
        if (found == initialized_tiles.end())
        {
            continue;
        }
        g_persisted_tiles_by_impl[impl_key] = found->second;
        for (const auto &tile_ptr : found->second)
        {
            if (tile_ptr != nullptr)
            {
                g_persisted_tile_pool.push_back(tile_ptr);
                ++tiles_retained;
            }
        }
        ++storages_mapped;
        if (is_context_verbose())
        {
            log_tile_adoption(
                "capture: impl=" +
                std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)) +
                " tensor='" + std::string(tensor_node_label(node)) +
                "' tiles=" + std::to_string(found->second.size()));
        }
    }
    if (is_context_verbose())
    {
        log_tile_adoption(
            "capture: initialized_tensors=" +
            std::to_string(initialized_tiles.size()) +
            " persistent_for_adoption=" + std::to_string(storages_mapped) +
            " tiles_retained=" + std::to_string(tiles_retained) +
            " pool_size=" + std::to_string(pool_before) + "->" +
            std::to_string(g_persisted_tile_pool.size()));
    }
}

void stage_persisted_tiles_for_session(
    nntile::Runtime &runtime,
    const nntile::TileGraph &tile_graph)
{
    if (g_persisted_tiles_by_impl.empty())
    {
        log_tile_adoption("stage: no persisted impls");
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
    for (const auto &[impl_key, tiles] : g_persisted_tiles_by_impl)
    {
        const auto mapped = g_tensor_nodes.find(impl_key);
        if (mapped == g_tensor_nodes.end())
        {
            log_tile_adoption(
                "stage: skip impl=" +
                std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)) +
                " (no pending tensor node)");
            continue;
        }
        nntile::TensorGraph::TensorNode const *node =
            bind_target_node(mapped->second);
        if (node == nullptr)
        {
            log_tile_adoption(
                "stage: skip impl=" +
                std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)) +
                " (no pending tensor node)");
            continue;
        }
        const auto tm_it = tile_map.find(node);
        if (tm_it == tile_map.end())
        {
            log_tile_adoption(
                "stage: skip impl=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(impl_key)) +
                " tensor='" + std::string(tensor_node_label(node)) +
                "' (not in new tile_map)");
            continue;
        }
        if (tm_it->second.size() != tiles.size())
        {
            log_tile_adoption(
                "stage: skip impl=" + std::to_string(
                    reinterpret_cast<std::uintptr_t>(impl_key)) +
                " tensor='" + std::string(tensor_node_label(node)) +
                "' (tile count mismatch saved=" +
                std::to_string(tiles.size()) + " new=" +
                std::to_string(tm_it->second.size()) + ")");
            continue;
        }
        log_tile_adoption(
            "stage: candidate impl=" + std::to_string(
                reinterpret_cast<std::uintptr_t>(impl_key)) +
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

void transfer_pinned_tensors_locked(std::vector<at::Tensor> &pin_drop)
{
    if (g_pinned_tensors.empty())
    {
        return;
    }
    pin_drop.insert(
        pin_drop.end(),
        std::make_move_iterator(g_pinned_tensors.begin()),
        std::make_move_iterator(g_pinned_tensors.end()));
    g_pinned_tensors.clear();
}

void clear_pending_graph_after_compile_locked(
    std::vector<at::Tensor> &pin_drop)
{
    g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    g_all_nodes.clear();
    for (auto it = g_tensor_nodes.begin(); it != g_tensor_nodes.end();)
    {
        if (!should_retain_mapped_tensor_after_compile(it->first, it->second))
        {
            it = g_tensor_nodes.erase(it);
            continue;
        }
        it->second.node = nullptr;
        it->second.staging_node = nullptr;
        if (is_staged_input_impl(it->first) || it->second.needs_host_copy ||
            it->second.is_persistent_input)
        {
            it->second.bind_at_execute = true;
        }
        ++it;
    }
    g_param_grad_nodes.clear();
    g_param_grad_registry.clear();
    g_relu_preactivation_stack.clear();
    transfer_pinned_tensors_locked(pin_drop);
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
    for (auto &[impl_key, mapped] : g_tensor_nodes)
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
            std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)));
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

void compile_graph_locked(
    bool clear_pending_after,
    std::vector<at::Tensor> &pin_drop)
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

    for (const auto &[impl_key, mapped] : g_tensor_nodes)
    {
        if (mapped.node == nullptr)
        {
            continue;
        }
        if (!mapped.bind_at_execute && !mapped.is_persistent_input)
        {
            continue;
        }
        if (mapped.host_data_ptr == nullptr)
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
                "compile: skip bind (already initialized / adopted) impl=" +
                std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)) +
                " tensor='" + std::string(tensor_node_label(bind_node)) +
                "'");
            continue;
        }
        log_tile_adoption(
            "compile: bind staging impl=" +
            std::to_string(reinterpret_cast<std::uintptr_t>(impl_key)) +
            " tensor='" + std::string(tensor_node_label(bind_node)) +
            "' nelems=" + std::to_string(mapped.count));
        bind_storage_to_runtime(
            *g_session->runtime,
            mapped.host_data_ptr,
            mapped);
    }

    for (const auto &[impl_key, mapped] : g_tensor_nodes)
    {
        nntile::TensorGraph::TensorNode *stored = mapped.node;
        if (stored == nullptr)
        {
            stored = bind_target_node(mapped);
        }
        if (stored != nullptr)
        {
            g_session->impl_to_node[impl_key] = stored;
        }
    }

    if (clear_pending_after)
    {
        clear_pending_graph_after_compile_locked(pin_drop);
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

void reset_recorder_locked(
    bool clear_tensor_gc,
    std::vector<at::Tensor> &pin_drop)
{
    g_graph.reset();
    g_tensor_nodes.clear();
    g_param_grad_nodes.clear();
    g_param_grad_registry.clear();
    g_relu_preactivation_stack.clear();
    g_all_nodes.clear();
    transfer_pinned_tensors_locked(pin_drop);
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
    g_session.reset();
    g_persisted_tiles_by_impl.clear();
    g_persisted_tile_pool.clear();
    drain_starpu_after_session_teardown();
    if (clear_tensor_gc)
    {
        clear_tensor_gc_state();
    }
}

void register_grad_alias_for_host_copy_locked(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node)
{
    if (!is_graph_mode() || grad_node == nullptr)
    {
        return;
    }
    if (!has_host_staging(grad))
    {
        ensure_host_staging(grad);
    }
    void *host_ptr = nullptr;
    if (has_host_staging(grad))
    {
        host_ptr = grad.storage().data_ptr().get();
    }
    g_tensor_nodes[tensor_impl_key(grad)] = MappedTensor{
        grad_node,
        nullptr,
        grad_node->dtype(),
        static_cast<std::size_t>(grad_node->nelems()),
        host_ptr != nullptr,
        false,
        is_staged_input_tensor(grad),
        host_ptr};
}

void sync_param_grad_aliases_locked()
{
    for (auto &[param_key, entry] : g_param_grad_registry)
    {
        (void) param_key;
        if (entry.grad_node == nullptr || !entry.param.defined())
        {
            continue;
        }
        const at::Tensor grad = entry.param.grad();
        if (!grad.defined())
        {
            continue;
        }
        at::Tensor mutable_grad = grad;
        register_grad_alias_for_host_copy_locked(
            mutable_grad,
            entry.grad_node);
    }
}

void execute_pending_graph_locked(std::vector<at::Tensor> &pin_drop)
{
    compile_graph_locked(false, pin_drop);
    run_graph_locked();
    sync_current_run_visible_outputs_locked();
    clear_pending_graph_after_compile_locked(pin_drop);
    if (!is_graph_mode())
    {
        reset_recorder_locked(false, pin_drop);
    }
}

void shutdown_recorder_locked(std::vector<at::Tensor> &pin_drop)
{
    if (g_graph != nullptr && g_graph->num_ops() > 0)
    {
        compile_graph_locked(false, pin_drop);
        run_graph_locked();
    }
    reset_recorder_locked(true, pin_drop);
}

} // namespace

bool has_pending_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    return g_graph != nullptr && g_graph->num_ops() > 0;
}

void require_no_pending_graph(const char *op_name)
{
    if (!is_graph_mode())
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
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
    std::vector<at::Tensor> pin_drop;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        execute_pending_graph_locked(pin_drop);
    }
}

void compile_graph()
{
    std::vector<at::Tensor> pin_drop;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        compile_graph_locked(true, pin_drop);
    }
}

void run_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    run_graph_locked();
}

void reset_graph_session()
{
    std::vector<at::Tensor> pin_drop;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        reset_recorder_locked(true, pin_drop);
    }
}

void shutdown_recorder()
{
    std::vector<at::Tensor> pin_drop;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        shutdown_recorder_locked(pin_drop);
    }
}

bool has_graph_session()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    return g_session != nullptr && g_session->runtime != nullptr;
}

nntile::TensorGraph::TensorNode *node_for_impl_locked(TensorImplKey impl_key)
{
    return session_node_for_impl_locked(impl_key);
}

void sync_nntile_storage_to_runtime(void *host_data_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    TensorImplKey impl_key = nullptr;
    for (const auto &[key, mapped] : g_tensor_nodes)
    {
        if (mapped.host_data_ptr == host_data_ptr)
        {
            impl_key = key;
            break;
        }
    }
    if (impl_key == nullptr)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *node = node_for_impl_locked(impl_key);
    if (node == nullptr)
    {
        return;
    }
    MappedTensor mapped;
    const auto found = g_tensor_nodes.find(impl_key);
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
        mapped.host_data_ptr = host_data_ptr;
    }
    bind_storage_to_runtime(*g_session->runtime, host_data_ptr, mapped);
}

void sync_runtime_to_nntile_storage(void *host_data_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    TensorImplKey impl_key = nullptr;
    for (const auto &[key, mapped] : g_tensor_nodes)
    {
        if (mapped.host_data_ptr == host_data_ptr)
        {
            impl_key = key;
            break;
        }
    }
    if (impl_key == nullptr)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *node = node_for_impl_locked(impl_key);
    if (node == nullptr)
    {
        return;
    }
    g_session->runtime->wait();
    MappedTensor mapped;
    mapped.node = node;
    mapped.dtype = node->dtype();
    mapped.count = static_cast<std::size_t>(node->nelems());
    mapped.host_data_ptr = host_data_ptr;
    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end())
    {
        mapped.needs_host_copy = found->second.needs_host_copy;
    }
    else
    {
        mapped.needs_host_copy = true;
    }
    copy_tensor_from_runtime(*g_session->runtime, mapped, host_data_ptr);
}

void sync_runtime_to_nntile_tensor(const at::Tensor &tensor)
{
    if (!has_host_staging(tensor))
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    nntile::TensorGraph::TensorNode *node =
        session_node_for_impl_locked(impl_key);
    if (node == nullptr)
    {
        return;
    }
    g_session->runtime->wait();
    void *host_data_ptr = tensor.storage().data_ptr().get();
    MappedTensor mapped;
    mapped.node = node;
    mapped.dtype = node->dtype();
    mapped.count = static_cast<std::size_t>(node->nelems());
    mapped.needs_host_copy = true;
    mapped.host_data_ptr = host_data_ptr;
    copy_tensor_from_runtime(*g_session->runtime, mapped, host_data_ptr);
}

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_session == nullptr || g_session->runtime == nullptr)
    {
        return;
    }
    const TensorImplKey impl_key = tensor_impl_key(src);
    nntile::TensorGraph::TensorNode *node = node_for_impl_locked(impl_key);
    if (node == nullptr)
    {
        return;
    }
    g_session->runtime->wait();
    MappedTensor mapped;
    mapped.node = node;
    mapped.dtype = node->dtype();
    mapped.count = static_cast<std::size_t>(node->nelems());
    mapped.needs_host_copy = true;
    mapped.host_data_ptr = dst.storage().data_ptr().get();
    copy_tensor_from_runtime(*g_session->runtime, mapped, mapped.host_data_ptr);
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
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    }
    return *g_graph;
}

nntile::TensorGraph::TensorNode *get_or_create_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    }

    const TensorImplKey impl_key = tensor_impl_key(tensor);
    void *host_ptr = nullptr;
    if (has_host_staging(tensor))
    {
        host_ptr = tensor.storage().data_ptr().get();
    }

    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end() && found->second.node != nullptr)
    {
        nntile::TensorGraph::TensorNode *existing = found->second.node;
        if (!shapes_equal(existing->shape(), shape))
        {
            if (graph_numel(existing->shape()) != graph_numel(shape))
            {
                g_tensor_nodes.erase(found);
            }
            else
            {
                existing = ensure_view_alias_locked(existing, shape, dtype);
                MappedTensor updated = found->second;
                updated.node = existing;
                updated.dtype = dtype;
                updated.count = static_cast<std::size_t>(graph_numel(shape));
                g_tensor_nodes[impl_key] = updated;
                MappedTensor &mapped = g_tensor_nodes[impl_key];
                if (!mapped.is_persistent_input && mapped.needs_host_copy)
                {
                    mapped.bind_at_execute = false;
                    mapped.needs_host_copy = false;
                }
                track_node(existing);
                return existing;
            }
        }
        else
        {
            if (!found->second.is_persistent_input &&
                found->second.needs_host_copy)
            {
                found->second.bind_at_execute = false;
                found->second.needs_host_copy = false;
            }
            track_node(existing);
            return existing;
        }
    }

    const auto found_after_stale = g_tensor_nodes.find(impl_key);
    const bool is_persistent =
        found_after_stale != g_tensor_nodes.end() &&
        found_after_stale->second.is_persistent_input;
    const bool staged = is_staged_input_tensor(tensor);
    const bool bind_at_execute =
        (found_after_stale != g_tensor_nodes.end() &&
            found_after_stale->second.bind_at_execute) ||
        mark_as_input || staged;

    auto *node = g_graph->data(shape, dtype);
    if (mark_as_input || is_persistent || staged)
    {
        node->mark_input(true);
    }
    apply_axis_name_hints_locked(impl_key, node);
    track_node(node);
    g_tensor_nodes[impl_key] = MappedTensor{
        node,
        nullptr,
        dtype,
        static_cast<std::size_t>(graph_numel(shape)),
        false,
        bind_at_execute,
        is_persistent || mark_as_input || staged,
        host_ptr};
    return node;
}

void register_data_node(
    const at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    node->mark_output(true);
    track_node(node);

    const TensorImplKey impl_key = tensor_impl_key(tensor);
    void *host_ptr = nullptr;
    if (has_host_staging(tensor))
    {
        host_ptr = tensor.storage().data_ptr().get();
    }
    const bool staged = is_staged_input_tensor(tensor);
    const bool metadata = is_metadata_only_tensor(tensor);
    const bool needs_host = !metadata && has_host_staging(tensor) &&
        (!is_graph_mode() || staged);

    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end() && found->second.is_persistent_input)
    {
        found->second.node = node;
        found->second.dtype = node->dtype();
        found->second.count = static_cast<std::size_t>(node->nelems());
        found->second.needs_host_copy = true;
        found->second.host_data_ptr = host_ptr;
        return;
    }

    g_tensor_nodes[impl_key] = MappedTensor{
        node,
        nullptr,
        node->dtype(),
        static_cast<std::size_t>(node->nelems()),
        needs_host,
        false,
        staged,
        host_ptr};
}

nntile::TensorGraph::TensorNode *lookup_data_node(TensorImplKey impl_key)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end())
    {
        return nullptr;
    }
    return found->second.node;
}

nntile::TensorGraph::TensorNode *lookup_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end() || found->second.node == nullptr)
    {
        return nullptr;
    }
    if (!graph_shape_matches_node(shape, found->second.node))
    {
        g_tensor_nodes.erase(found);
        return nullptr;
    }
    return found->second.node;
}

void register_param_grad_node(
    const at::Tensor &param,
    nntile::TensorGraph::TensorNode *grad_node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey key = tensor_impl_key(param);
    g_param_grad_nodes[key] = grad_node;
    g_param_grad_registry[key] = ParamGradEntry{grad_node, param};
}

nntile::TensorGraph::TensorNode *lookup_param_grad_node(
    const at::Tensor &param)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const auto found = g_param_grad_nodes.find(tensor_impl_key(param));
    if (found == g_param_grad_nodes.end())
    {
        return nullptr;
    }
    return found->second;
}

void register_grad_alias_for_host_copy(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    register_grad_alias_for_host_copy_locked(grad, grad_node);
}

void push_relu_preactivation_node(nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    g_relu_preactivation_stack.push_back(node);
}

nntile::TensorGraph::TensorNode *pop_relu_preactivation_node(
    const std::vector<nntile::Index> &shape)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    for (auto it = g_relu_preactivation_stack.rbegin();
         it != g_relu_preactivation_stack.rend();
         ++it)
    {
        if (!graph_shape_matches_node(shape, *it))
        {
            continue;
        }
        nntile::TensorGraph::TensorNode *node = *it;
        g_relu_preactivation_stack.erase(std::next(it).base());
        return node;
    }
    return nullptr;
}

void on_tensor_impl_released(TensorImplKey key)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    for (const at::Tensor &pinned : g_pinned_tensors)
    {
        if (tensor_impl_key(pinned) == key)
        {
            return;
        }
    }
    g_tensor_nodes.erase(key);
    g_persisted_tiles_by_impl.erase(key);
    g_param_grad_nodes.erase(key);
    g_param_grad_registry.erase(key);
    g_axis_name_hints.erase(key);
}

void record_view_alias(const at::Tensor &self, const at::Tensor &view)
{
    if (!is_graph_mode())
    {
        return;
    }
    if (view.device().type() != c10::DeviceType::PrivateUse1)
    {
        return;
    }
    std::vector<nntile::Index> view_shape;
    view_shape.reserve(static_cast<std::size_t>(view.dim()));
    for (const auto dim : view.sizes())
    {
        view_shape.push_back(static_cast<nntile::Index>(dim));
    }
    const TensorImplKey view_key = tensor_impl_key(view);
    const TensorImplKey self_key = tensor_impl_key(self);

    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        return;
    }

    const MappedTensor *source = nullptr;
    const auto view_it = g_tensor_nodes.find(view_key);
    if (view_it != g_tensor_nodes.end() && view_it->second.node != nullptr)
    {
        source = &view_it->second;
    }
    else
    {
        const auto self_it = g_tensor_nodes.find(self_key);
        if (self_it != g_tensor_nodes.end() && self_it->second.node != nullptr)
        {
            source = &self_it->second;
        }
    }
    if (source == nullptr)
    {
        return;
    }

    const nntile::DataType dtype = source->dtype;
    nntile::TensorGraph::TensorNode *const src_node = source->node;
    if (shapes_equal(src_node->shape(), view_shape))
    {
        if (view_key != self_key)
        {
            g_tensor_nodes[view_key] = *source;
        }
        return;
    }
    nntile::TensorGraph::TensorNode *view_node = ensure_view_alias_locked(
        src_node,
        view_shape,
        dtype);
    MappedTensor updated = *source;
    updated.node = view_node;
    updated.count = static_cast<std::size_t>(graph_numel(view_shape));
    g_tensor_nodes[view_key] = updated;
}

void track_graph_node(nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    track_node(node);
}

void pin_tensor_for_graph(const at::Tensor &tensor)
{
    if (!is_graph_mode())
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    g_pinned_tensors.push_back(tensor);
    if (const char *env = std::getenv("TORCH_NNTILE_TRACE_STORAGE");
        env != nullptr && env[0] != '\0' && env[0] != '0')
    {
        std::cerr << "[torch_nntile pin] data_ptr="
                  << tensor.storage().data_ptr().get()
                  << " pinned_count=" << g_pinned_tensors.size() << '\n';
    }
}

void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs)
{
    if (!is_graph_mode())
    {
        return;
    }
    for (const at::Tensor &tensor : inputs)
    {
        if (is_staged_input_tensor(tensor) || has_host_staging(tensor))
        {
            pin_tensor_for_graph(tensor);
        }
    }
}

void pin_graph_op_output(const at::Tensor &output, bool pin_output)
{
    if (!is_graph_mode() || !pin_output)
    {
        return;
    }
    if (has_host_staging(output))
    {
        pin_tensor_for_graph(output);
    }
}

void set_axis_group_name(
    TensorImplKey impl_key,
    int ndim,
    const std::unordered_map<int, std::string> &names)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
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
        const auto found = g_tensor_nodes.find(impl_key);
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
            g_axis_name_hints[impl_key][dim] = name;
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
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
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
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
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

GcDebugStats debug_gc_stats()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    GcDebugStats stats;
    stats.pinned_tensors = static_cast<std::int64_t>(g_pinned_tensors.size());
    stats.tensor_nodes = static_cast<std::int64_t>(g_tensor_nodes.size());
    stats.tile_pool = static_cast<std::int64_t>(g_persisted_tile_pool.size());
    if (g_graph != nullptr)
    {
        stats.pending_ops = static_cast<std::int64_t>(g_graph->num_ops());
        stats.pending_data = static_cast<std::int64_t>(g_graph->num_data());
    }
    stats.has_session =
        g_session != nullptr && g_session->runtime != nullptr;
    return stats;
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

void sync_runtime_to_nntile_tensor(const at::Tensor & /*tensor*/)
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

void on_tensor_impl_released(TensorImplKey /*key*/)
{
}

void record_view_alias(const at::Tensor & /*self*/, const at::Tensor & /*view*/)
{
}

void set_axis_group_name(
    TensorImplKey /*impl_key*/,
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

GcDebugStats debug_gc_stats()
{
    return {};
}

void shutdown_recorder()
{
}

void copy_nntile_tensor_to_cpu(const at::Tensor & /*src*/, at::Tensor & /*dst*/)
{
}

} // namespace torch_nntile

#endif
