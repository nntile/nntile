/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_graph_recorder.cpp
 */

#include "nntile_graph_recorder.h"
#include "nntile_graph_recorder_impl.h"
#include "nntile_tensor_gc.h"
#include "nntile_tensor_meta.h"

#include "nntile_context.h"

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <cstring>
#include <stdexcept>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <nntile/runtime.hh>
#include <nntile/dtype.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/gather.hh>
#include <nntile/tensor/ops/scatter.hh>
#include <nntile/tensor/ops/contiguous_view.hh>
#include <nntile/tensor/tensor_graph_tiling.hh>
#include <nntile/tile/append_tensor_graph_phase.hh>
#include <nntile/tile/graph.hh>
#include <nntile/tile/lower_staging_tensor.hh>

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

std::recursive_mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
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
std::unordered_set<nntile::TensorGraph::TensorNode *> g_invalidated_stagings;

struct RecorderExecState
{
    std::unique_ptr<nntile::TileGraph> tile_graph;
    std::unique_ptr<nntile::Runtime> runtime;
    nntile::TileGraphIncrementalState inc_state;
    nntile::TensorNodeToTileMap tile_map;
    //! Pins transferred at compile; kept alive until run_graph() finishes.
    std::vector<at::Tensor> pin_hold;
    //! Post-DCE execution_order index already submitted via execute_range.
    std::size_t executed_op_end = 0;
    //! Slice scheduled by the latest compile_graph_locked call.
    std::size_t pending_exec_op_begin = 0;
    std::size_t pending_exec_op_end = 0;
    //! Scatter staging tensors in the pending phase (invalidate after run).
    std::vector<nntile::TensorGraph::TensorNode *> pending_scatter_stagings;
};

std::unique_ptr<RecorderExecState> g_exec;
bool g_defer_pending_clear_after_run = false;

void sync_param_grad_aliases_locked();

void compile_graph_locked(
    bool clear_pending_after,
    std::vector<at::Tensor> &pin_drop);

void run_graph_locked();

void register_grad_alias_for_host_copy_locked(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node);

void clear_param_grad_registry_locked()
{
    std::vector<at::Tensor> params_to_release;
    params_to_release.reserve(g_param_grad_registry.size());
    for (auto &[key, entry] : g_param_grad_registry)
    {
        (void) key;
        if (entry.param.defined())
        {
            params_to_release.push_back(std::move(entry.param));
        }
    }
    g_param_grad_registry.clear();
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

nntile::DataType aten_scalar_to_nntile_dtype(at::ScalarType dtype)
{
    switch (dtype)
    {
    case at::ScalarType::Float:
        return nntile::DataType::FP32;
    case at::ScalarType::Long:
        return nntile::DataType::INT64;
    case at::ScalarType::Bool:
        return nntile::DataType::BOOL;
    case at::ScalarType::Byte:
        return nntile::DataType::BOOL;
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported nntile tensor dtype");
    }
}

std::vector<nntile::Index> aten_sizes_to_graph_shape(at::IntArrayRef sizes)
{
    std::vector<nntile::Index> shape;
    shape.reserve(sizes.size());
    for (const auto dim : sizes)
    {
        shape.push_back(static_cast<nntile::Index>(dim));
    }
    return shape;
}

void ensure_recorder_exec_state_locked()
{
    ensure_nntile_context();
    if (g_exec == nullptr)
    {
        g_exec = std::make_unique<RecorderExecState>();
    }
    if (g_exec->tile_graph == nullptr)
    {
        g_exec->tile_graph =
            std::make_unique<nntile::TileGraph>("torch_nntile_tile");
    }
    if (g_exec->runtime == nullptr)
    {
        g_exec->runtime =
            std::make_unique<nntile::Runtime>(*g_exec->tile_graph);
    }
}

void compile_exec_runtime_locked()
{
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    g_exec->runtime->compile();
}

void lower_io_staging_locked(nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr || g_graph == nullptr)
    {
        return;
    }
    ensure_recorder_exec_state_locked();
    const nntile::TensorGraphTiling tiling =
        nntile::TensorGraphTiling::from_tensor_graph(*g_graph);
    nntile::lower_staging_tensor_immediate(
        *g_graph,
        staging,
        tiling,
        *g_exec->tile_graph,
        g_exec->inc_state,
        g_exec->tile_map);
    compile_exec_runtime_locked();
}

nntile::TileGraph::TileNode *require_single_staging_tile_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    if (g_exec == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: no runtime for io_staging tile lookup");
    }
    const auto found = g_exec->tile_map.find(staging);
    if (found == g_exec->tile_map.end() || found->second.size() != 1)
    {
        throw std::runtime_error(
            "torch_nntile: io_staging must be single-tile");
    }
    nntile::TileGraph::TileNode *tile = found->second[0];
    if (tile == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: io_staging tile missing");
    }
    return tile;
}

void write_cpu_bytes_to_staging_locked(
    nntile::TensorGraph::TensorNode *staging,
    const void *host_ptr,
    nntile::DataType dtype,
    std::size_t count)
{
    if (staging == nullptr || host_ptr == nullptr || count == 0)
    {
        return;
    }
    ensure_recorder_exec_state_locked();
    nntile::Runtime &runtime = *g_exec->runtime;
    nntile::TileGraph::TileNode *tile =
        require_single_staging_tile_locked(staging);
    switch (dtype)
    {
    case nntile::DataType::FP32:
    {
        auto &buf = runtime.get_tile<nntile::fp32_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging write size mismatch");
        }
        auto local = buf.acquire(STARPU_W);
        const auto *src = static_cast<const float *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            local[static_cast<nntile::Index>(i)] = nntile::fp32_t(src[i]);
        }
        local.release();
        break;
    }
    case nntile::DataType::INT64:
    {
        auto &buf = runtime.get_tile<nntile::int64_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging write size mismatch");
        }
        auto local = buf.acquire(STARPU_W);
        const auto *src = static_cast<const std::int64_t *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            local[static_cast<nntile::Index>(i)] =
                nntile::int64_t(src[i]);
        }
        local.release();
        break;
    }
    case nntile::DataType::BOOL:
    {
        auto &buf = runtime.get_tile<nntile::bool_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging write size mismatch");
        }
        auto local = buf.acquire(STARPU_W);
        const auto *src = static_cast<const bool *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            local[static_cast<nntile::Index>(i)] = nntile::bool_t(src[i]);
        }
        local.release();
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging write dtype");
    }
    runtime.mark_initialized(staging);
    g_invalidated_stagings.erase(staging);
}

void invalidate_staging_tile_buffer_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr || g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    g_invalidated_stagings.insert(staging);
    g_exec->runtime->invalidate_initialized(staging);
}

void read_staging_to_host_locked(
    nntile::TensorGraph::TensorNode *staging,
    void *host_ptr,
    nntile::DataType dtype,
    std::size_t count)
{
    if (staging == nullptr || host_ptr == nullptr || count == 0)
    {
        return;
    }
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: no runtime for staging readout");
    }
    nntile::Runtime &runtime = *g_exec->runtime;
    runtime.wait();
    nntile::TileGraph::TileNode *tile =
        require_single_staging_tile_locked(staging);
    switch (dtype)
    {
    case nntile::DataType::FP32:
    {
        const auto &buf = runtime.get_tile<nntile::fp32_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        auto local = buf.acquire(STARPU_R);
        auto *dst = static_cast<float *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<float>(local[static_cast<nntile::Index>(i)]);
        }
        local.release();
        break;
    }
    case nntile::DataType::INT64:
    {
        const auto &buf = runtime.get_tile<nntile::int64_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        auto local = buf.acquire(STARPU_R);
        auto *dst = static_cast<std::int64_t *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<std::int64_t>(
                local[static_cast<nntile::Index>(i)]);
        }
        local.release();
        break;
    }
    case nntile::DataType::BOOL:
    {
        const auto &buf = runtime.get_tile<nntile::bool_t>(tile);
        if (count != static_cast<std::size_t>(buf.nelems))
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        auto local = buf.acquire(STARPU_R);
        auto *dst = static_cast<bool *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<bool>(local[static_cast<nntile::Index>(i)]);
        }
        local.release();
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging read dtype");
    }
}

nntile::TensorGraph::TensorNode *ensure_io_staging_node_locked(
    NodeRef binding)
{
    if (binding == nullptr || binding->logical == nullptr || g_graph == nullptr)
    {
        return nullptr;
    }
    if (binding->io_staging != nullptr)
    {
        return binding->io_staging;
    }
    auto *staging = g_graph->data(
        binding->logical->shape(),
        binding->logical->dtype());
    staging->mark_input(true);
    staging->mark_output(false);
    staging->set_name(
        std::string("io_staging_") + binding->logical->name());
    track_node(staging);
    binding->io_staging = staging;
    lower_io_staging_locked(staging);
    return staging;
}

bool should_pin_tensor_for_graph_locked(const at::Tensor &tensor)
{
    if (nntile_io_staging(tensor) != nullptr)
    {
        return true;
    }
    return is_metadata_only_tensor(tensor) &&
        nntile_binding(tensor) != nullptr;
}

void pin_tensor_for_graph(const at::Tensor &tensor)
{
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

bool staging_ready_for_direct_read_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr ||
        g_invalidated_stagings.count(staging) != 0)
    {
        return false;
    }
    return g_exec != nullptr &&
        g_exec->runtime != nullptr && g_exec->runtime->is_compiled() &&
        g_exec->runtime->is_initialized(staging);
}

bool can_read_tensor_from_staging_locked(const at::Tensor &tensor)
{
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr || binding->io_staging == nullptr)
    {
        return false;
    }
    return staging_ready_for_direct_read_locked(binding->io_staging);
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

nntile::TensorGraph::TensorNode *ensure_graph_shape_bridge_locked(
    nntile::TensorGraph::TensorNode *node,
    const std::vector<nntile::Index> &shape)
{
    if (node == nullptr)
    {
        return nullptr;
    }
    if (shapes_equal(node->shape(), shape))
    {
        return node;
    }
    if (graph_numel(node->shape()) != graph_numel(shape))
    {
        return node;
    }
    auto *view_node = g_graph->data(shape, node->dtype());
    track_node(view_node);
    nntile::tensor::contiguous_view(node, view_node);
    return view_node;
}

nntile::TensorGraph::TensorNode *logical_node_for_tensor_locked(
    at::Tensor &mutable_tensor,
    const TensorImplKey &impl_key,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input)
{
    if (NodeRef binding = nntile_binding(mutable_tensor);
        binding != nullptr && binding->logical != nullptr)
    {
        nntile::TensorGraph::TensorNode *logical = binding->logical;
        if (logical->graph() != g_graph.get())
        {
            throw std::runtime_error(
                "torch_nntile: tensor logical node does not belong to the "
                "active TensorGraph");
        }
        if (graph_numel(logical->shape()) != graph_numel(shape))
        {
            throw std::invalid_argument(
                "torch_nntile: logical node numel mismatch for tensor");
        }
        nntile::TensorGraph::TensorNode *node =
            ensure_graph_shape_bridge_locked(logical, shape);
        return node;
    }

    auto *node = g_graph->data(shape, dtype);
    if (mark_as_input)
    {
        node->mark_input(true);
    }
    apply_axis_name_hints_locked(impl_key, node);
    track_node(node);

    if (nntile_binding(mutable_tensor) == nullptr)
    {
        auto new_binding = std::make_shared<NNTileBinding>(node);
        attach_binding(mutable_tensor, new_binding);
    }

    return node;
}

void sync_current_run_visible_outputs_locked()
{
    (void)0;
    // Phase 7: outputs are read via gather + staging, not host Storage.
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
    g_param_grad_nodes.clear();
    clear_param_grad_registry_locked();
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
    starpu_task_wait_for_all();
}

void insert_input_scatter_staging_locked()
{
    // Phase 7: scatter is recorded at .to("nntile") time, not at compile.
}

void collect_scatter_stagings_from_phase_locked(
    const nntile::TensorGraph::PhaseSnapshot &phase,
    std::vector<nntile::TensorGraph::TensorNode *> &out)
{
    if (g_graph == nullptr || phase.empty())
    {
        return;
    }
    const auto &ops = g_graph->ops();
    for (std::size_t i = phase.op_begin; i < phase.op_end; ++i)
    {
        if (i >= ops.size() || ops[i] == nullptr)
        {
            continue;
        }
        if (ops[i]->op_name() != "SCATTER")
        {
            continue;
        }
        const auto *scatter =
            dynamic_cast<const nntile::tensor::TensorScatterOp *>(
                ops[i].get());
        if (scatter != nullptr && scatter->src != nullptr)
        {
            out.push_back(scatter->src);
        }
    }
}

void compile_graph_locked(
    bool clear_pending_after,
    std::vector<at::Tensor> &pin_drop)
{
    if (g_graph == nullptr ||
        g_graph->num_ops() <= g_graph->phase_seal_cursor())
    {
        return;
    }

    ensure_nntile_context();
    ensure_recorder_exec_state_locked();

    sync_param_grad_aliases_locked();
    apply_pending_axis_tiling_locked();

    const nntile::TensorGraph::PhaseSnapshot phase = g_graph->seal_phase();
    std::vector<nntile::TensorGraph::TensorNode *> scatter_stagings;
    collect_scatter_stagings_from_phase_locked(phase, scatter_stagings);
    const nntile::TensorGraphTiling tiling =
        nntile::TensorGraphTiling::from_tensor_graph(*g_graph);

    nntile::append_tensor_graph_phase(
        *g_graph,
        phase,
        tiling,
        *g_exec->tile_graph,
        g_exec->inc_state,
        g_exec->tile_map);

    g_exec->pending_exec_op_begin = g_exec->executed_op_end;
    g_exec->runtime->compile();
    g_exec->pending_exec_op_end =
        g_exec->runtime->execution_op_count();
    g_exec->pending_scatter_stagings = std::move(scatter_stagings);

    if (clear_pending_after)
    {
        transfer_pinned_tensors_locked(pin_drop);
        g_defer_pending_clear_after_run = true;
    }
}

void run_graph_locked()
{
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    if (g_exec->pending_exec_op_end > g_exec->pending_exec_op_begin)
    {
        g_exec->runtime->execute_range(
            g_exec->pending_exec_op_begin,
            g_exec->pending_exec_op_end);
        g_exec->runtime->wait();
        g_exec->pending_scatter_stagings.clear();
        g_exec->executed_op_end = g_exec->pending_exec_op_end;
    }
    else
    {
        g_exec->runtime->wait();
    }
    if (g_defer_pending_clear_after_run)
    {
        std::vector<at::Tensor> pin_drop;
        clear_pending_graph_after_compile_locked(pin_drop);
        g_defer_pending_clear_after_run = false;
    }
}

void reset_recorder_locked(
    bool clear_tensor_gc,
    std::vector<at::Tensor> &pin_drop)
{
    g_graph.reset();
    g_param_grad_nodes.clear();
    clear_param_grad_registry_locked();
    g_relu_preactivation_stack.clear();
    g_all_nodes.clear();
    transfer_pinned_tensors_locked(pin_drop);
    g_defer_pending_clear_after_run = false;
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
    g_invalidated_stagings.clear();
    g_exec.reset();
    drain_starpu_after_session_teardown();
    if (clear_tensor_gc)
    {
        clear_tensor_gc_state();
        clear_binding_registry();
    }
}

void register_grad_alias_for_host_copy_locked(
    at::Tensor &grad,
    nntile::TensorGraph::TensorNode *grad_node)
{
    if (grad_node == nullptr)
    {
        return;
    }
    if (nntile_binding(grad) == nullptr)
    {
        attach_binding(grad, std::make_shared<NNTileBinding>(grad_node));
    }
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
}

void shutdown_recorder_locked(std::vector<at::Tensor> &pin_drop)
{
    g_graph.reset();
    g_defer_pending_clear_after_run = false;
    reset_recorder_locked(true, pin_drop);
}

} // namespace

bool can_read_nntile_tensor_from_staging(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    return can_read_tensor_from_staging_locked(tensor);
}

bool read_nntile_staging_to_host(const at::Tensor &tensor, void *host_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (host_ptr == nullptr || !can_read_tensor_from_staging_locked(tensor))
    {
        return false;
    }
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr || binding->io_staging == nullptr ||
        binding->logical == nullptr)
    {
        return false;
    }
    read_staging_to_host_locked(
        binding->io_staging,
        host_ptr,
        binding->logical->dtype(),
        static_cast<std::size_t>(binding->logical->nelems()));
    return true;
}

bool has_pending_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    return g_graph != nullptr &&
        g_graph->num_ops() > g_graph->phase_seal_cursor();
}

void require_no_pending_graph(const char *op_name)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph != nullptr &&
        g_graph->num_ops() > g_graph->phase_seal_cursor())
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
        if (g_exec != nullptr && !pin_drop.empty())
        {
            g_exec->pin_hold = std::move(pin_drop);
        }
    }
}

void run_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    run_graph_locked();
    sync_current_run_visible_outputs_locked();
    if (g_exec != nullptr)
    {
        g_exec->pin_hold.clear();
    }
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
    return g_exec != nullptr && g_exec->runtime != nullptr;
}

nntile::TensorGraph::TensorNode *node_for_impl_locked(TensorImplKey impl_key)
{
    for (const at::Tensor &tensor : g_pinned_tensors)
    {
        if (tensor_impl_key(tensor) != impl_key)
        {
            continue;
        }
        if (nntile::TensorGraph::TensorNode *node = nntile_node(tensor);
            node != nullptr)
        {
            return node;
        }
    }
    return nullptr;
}

void gather_logical_to_staging_and_read_locked(
    nntile::TensorGraph::TensorNode *logical,
    nntile::TensorGraph::TensorNode *staging,
    void *host_ptr,
    nntile::DataType dtype,
    std::size_t count)
{
    if (logical == nullptr || staging == nullptr || host_ptr == nullptr ||
        count == 0)
    {
        return;
    }
    if (g_graph == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: gather readout requires an active TensorGraph");
    }
    const bool staging_was_output = staging->is_output();
    staging->mark_output(true);
    nntile::tensor::clear(staging);
    nntile::tensor::gather(logical, staging);

    std::vector<at::Tensor> pin_drop;
    compile_graph_locked(false, pin_drop);
    run_graph_locked();

    g_invalidated_stagings.erase(staging);
    g_exec->runtime->mark_initialized(staging);
    read_staging_to_host_locked(staging, host_ptr, dtype, count);
    invalidate_staging_tile_buffer_locked(staging);
    staging->mark_output(staging_was_output);
}

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    NodeRef binding = nntile_binding(src);
    if (binding == nullptr || binding->logical == nullptr)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *logical = binding->logical;
    const nntile::DataType dtype = logical->dtype();
    const std::size_t count =
        static_cast<std::size_t>(logical->nelems());
    void *host_ptr = dst.storage().data_ptr().get();

    // Pre-first-compile: bound staging still holds host bytes from .to().
    if (g_graph != nullptr && g_graph->phase_seal_cursor() == 0 &&
        can_read_tensor_from_staging_locked(src))
    {
        read_staging_to_host_locked(
            binding->io_staging,
            host_ptr,
            dtype,
            count);
        return;
    }

    if (g_exec == nullptr || g_exec->runtime == nullptr ||
        !g_exec->runtime->is_compiled() || g_exec->executed_op_end == 0)
    {
        throw std::runtime_error(
            "torch_nntile: copy nntile tensor to CPU requires compile_graph() "
            "and run() first");
    }

    nntile::TensorGraph::TensorNode *staging =
        ensure_io_staging_node_locked(binding);
    if (staging == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: copy nntile tensor to CPU missing io_staging");
    }
    gather_logical_to_staging_and_read_locked(
        logical,
        staging,
        host_ptr,
        dtype,
        count);
}

void init_nntile_input_from_cpu(
    const at::Tensor &cpu_src,
    at::Tensor &nntile_dst)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    TORCH_CHECK(cpu_src.is_cpu(), "init_nntile_input_from_cpu: expected CPU src");
    TORCH_CHECK(
        nntile_dst.device().type() == c10::DeviceType::PrivateUse1,
        "init_nntile_input_from_cpu: expected nntile dst");
    TORCH_CHECK(
        cpu_src.sizes() == nntile_dst.sizes(),
        "init_nntile_input_from_cpu: shape mismatch");
    TORCH_CHECK(
        cpu_src.is_contiguous() && nntile_dst.is_contiguous(),
        "init_nntile_input_from_cpu: contiguous tensors required");

    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
    }

    const std::vector<nntile::Index> shape =
        aten_sizes_to_graph_shape(cpu_src.sizes());
    const nntile::DataType dtype =
        aten_scalar_to_nntile_dtype(cpu_src.scalar_type());
    const TensorImplKey impl_key = tensor_impl_key(nntile_dst);

    if (NodeRef existing = nntile_binding(nntile_dst);
        existing != nullptr && existing->logical != nullptr)
    {
        auto *logical = existing->logical;
        auto *staging = existing->io_staging;
        if (staging == nullptr)
        {
            staging = ensure_io_staging_node_locked(existing);
        }
        staging->mark_input(true);
        write_cpu_bytes_to_staging_locked(
            staging,
            cpu_src.storage().data_ptr().get(),
            dtype,
            static_cast<std::size_t>(cpu_src.numel()));
        nntile::tensor::scatter(staging, logical);
        g_pinned_tensors.push_back(nntile_dst);
        return;
    }

    auto *logical = g_graph->data(shape, dtype);
    apply_axis_name_hints_locked(impl_key, logical);
    track_node(logical);

    auto binding = std::make_shared<NNTileBinding>(logical);
    auto *staging = g_graph->data(shape, dtype);
    staging->mark_input(true);
    staging->mark_output(false);
    staging->set_name(std::string("io_staging_") + logical->name());
    track_node(staging);
    binding->io_staging = staging;
    attach_binding(nntile_dst, binding);

    lower_io_staging_locked(staging);
    write_cpu_bytes_to_staging_locked(
        staging,
        cpu_src.storage().data_ptr().get(),
        dtype,
        static_cast<std::size_t>(cpu_src.numel()));

    nntile::tensor::scatter(staging, logical);

    g_pinned_tensors.push_back(nntile_dst);
}

void maybe_execute_after_record()
{
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
    at::Tensor mutable_tensor = const_cast<at::Tensor &>(tensor);
    nntile::TensorGraph::TensorNode *node = logical_node_for_tensor_locked(
        mutable_tensor,
        impl_key,
        shape,
        dtype,
        mark_as_input);
    assert_has_node_ref(tensor, "get_or_create_data_node");
    return node;
}

void register_data_node(
    const at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    track_node(node);
    at::Tensor mutable_tensor = tensor;
    if (nntile_binding(mutable_tensor) == nullptr)
    {
        attach_binding(
            mutable_tensor,
            std::make_shared<NNTileBinding>(node));
    }
    assert_has_node_ref(tensor, "register_data_node");
}

nntile::TensorGraph::TensorNode *lookup_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        return nullptr;
    }
    if (nntile::TensorGraph::TensorNode *bound = nntile_node(tensor);
        bound != nullptr &&
        bound->graph() == g_graph.get() &&
        graph_numel(bound->shape()) == graph_numel(shape))
    {
        return ensure_graph_shape_bridge_locked(bound, shape);
    }
    return nullptr;
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
    pin_tensor_for_graph(grad);
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
    unregister_binding_impl(key);
    if (g_param_grad_nodes.count(key) != 0)
    {
        g_param_grad_nodes.erase(key);
        g_param_grad_registry.erase(key);
    }
    g_axis_name_hints.erase(key);
}

void record_view_alias(const at::Tensor &self, const at::Tensor &view)
{
    if (view.device().type() != c10::DeviceType::PrivateUse1)
    {
        return;
    }
    nntile::TensorGraph::TensorNode *src_node = nntile_node(self);
    if (src_node == nullptr)
    {
        return;
    }
    std::vector<nntile::Index> view_shape;
    view_shape.reserve(static_cast<std::size_t>(view.dim()));
    for (const auto dim : view.sizes())
    {
        view_shape.push_back(static_cast<nntile::Index>(dim));
    }
    if (graph_numel(src_node->shape()) != graph_numel(view_shape))
    {
        throw std::invalid_argument(
            "view: storage alias must preserve numel");
    }
    at::Tensor mutable_view = view;
    share_node_ref_for_reshape(self, mutable_view);
    assert_has_node_ref(view, "record_view_alias");
}

void track_graph_node(nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    track_node(node);
}

void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    for (const at::Tensor &tensor : inputs)
    {
        if (should_pin_tensor_for_graph_locked(tensor))
        {
            pin_tensor_for_graph(tensor);
        }
    }
}

void pin_graph_op_output(const at::Tensor &output, bool pin_output)
{
    if (!pin_output)
    {
        return;
    }
    pin_tensor_for_graph(output);
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
        for (const at::Tensor &tensor : g_pinned_tensors)
        {
            if (tensor_impl_key(tensor) != impl_key)
            {
                continue;
            }
            node = nntile_node(tensor);
            if (node != nullptr)
            {
                break;
            }
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

bool is_tensor_graph_output(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (nntile::TensorGraph::TensorNode *node = nntile_node(tensor);
        node != nullptr)
    {
        return node->is_output();
    }
    return false;
}

void stage_tensor_for_axis_group_compile(const at::Tensor &tensor)
{
    if (is_tensor_graph_output(tensor))
    {
        return;
    }
    pin_tensor_for_graph(tensor);
}

void mark_persistent_graph_tensor(const at::Tensor &tensor)
{
    pin_tensor_for_graph(tensor);
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
    stats.live_bindings = count_live_bindings();
    stats.tile_pool = 0;
    if (g_graph != nullptr)
    {
        stats.pending_ops = static_cast<std::int64_t>(g_graph->num_ops());
        stats.pending_data = static_cast<std::int64_t>(g_graph->num_data());
    }
    stats.has_session =
        g_exec != nullptr && g_exec->runtime != nullptr;
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

bool is_tensor_graph_output(const at::Tensor & /*tensor*/)
{
    require_libnntile();
    return false;
}

void stage_tensor_for_axis_group_compile(const at::Tensor & /*tensor*/)
{
    require_libnntile();
}

void mark_persistent_graph_tensor(const at::Tensor & /*tensor*/)
{
}

bool read_nntile_staging_to_host(const at::Tensor & /*tensor*/, void * /*host_ptr*/)
{
    return false;
}

bool can_read_nntile_tensor_from_staging(const at::Tensor & /*tensor*/)
{
    return false;
}

void init_nntile_input_from_cpu(
    const at::Tensor &cpu_src,
    at::Tensor &nntile_dst)
{
    TORCH_CHECK(cpu_src.is_cpu(), "init_nntile_input_from_cpu: expected CPU src");
    TORCH_CHECK(
        nntile_dst.device().type() == c10::DeviceType::PrivateUse1,
        "init_nntile_input_from_cpu: expected nntile dst");
    TORCH_CHECK(
        cpu_src.sizes() == nntile_dst.sizes(),
        "init_nntile_input_from_cpu: shape mismatch");
    TORCH_CHECK(
        cpu_src.is_contiguous() && nntile_dst.is_contiguous(),
        "init_nntile_input_from_cpu: contiguous tensors required");
    ensure_host_staging(nntile_dst);
    const int64_t nbytes = cpu_src.nbytes();
    if (nbytes > 0)
    {
        std::memcpy(
            nntile_dst.data_ptr(),
            cpu_src.data_ptr(),
            static_cast<std::size_t>(nbytes));
    }
    mark_staged_input_tensor(nntile_dst);
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
