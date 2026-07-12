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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch_nntile
{

namespace
{

std::recursive_mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
struct ParamGradEntry
{
    nntile::TensorGraph::TensorNode *grad_node = nullptr;
    at::Tensor param;
};
std::unordered_map<TensorImplKey, ParamGradEntry> g_param_grad_registry;
std::vector<nntile::TensorGraph::TensorNode *> g_relu_preactivation_stack;
std::vector<at::Tensor> g_pinned_tensors;
std::unordered_map<TensorImplKey, std::unordered_map<int, std::string>>
    g_axis_name_hints;
std::unordered_map<std::string, std::vector<nntile::Index>> g_axis_tiling_by_name;
std::size_t g_ephemeral_staging_serial = 0;

struct RecorderExecState
{
    std::unique_ptr<nntile::TileGraph> tile_graph;
    std::unique_ptr<nntile::Runtime> runtime;
    nntile::TileGraphIncrementalState inc_state;
    nntile::TensorNodeToTileMap tile_map;
    //! Session-scoped layouts; ensure_phase_layouts only adds new tensors.
    std::shared_ptr<nntile::TensorGraphTiling> session_tiling;
    //! Pins transferred at compile; kept alive until wait_graph_session().
    std::vector<at::Tensor> pin_hold;
    //! Post-DCE execution_order index already submitted via execute_range.
    std::size_t executed_op_end = 0;
    //! Slice scheduled by the latest compile_graph_locked call.
    std::size_t pending_exec_op_begin = 0;
    std::size_t pending_exec_op_end = 0;
    //! Scatter staging tensors in the pending phase (invalidate after wait).
    std::vector<nntile::TensorGraph::TensorNode *> pending_scatter_stagings;
    //! Logical tensors that were mark_output(true) when the pending slice
    //! was compiled. After wait() drops step temps (pin_hold / NodeRef), any
    //! entry that is no longer marked is invalidated once.
    std::vector<nntile::TensorGraph::TensorNode *> pending_output_reclaim;
};

std::unique_ptr<RecorderExecState> g_exec;
bool g_defer_pending_clear_after_run = false;
//! True after run_graph() until wait_graph_session() finishes post-run work.
bool g_run_cleanup_pending = false;

struct GraphApiTimingStats
{
    std::uint64_t compile_calls = 0;
    double compile_s = 0.0;
    std::uint64_t compile_ops = 0;
    double compile_seal_s = 0.0;
    double compile_tiling_s = 0.0;
    double compile_append_s = 0.0;
    double compile_runtime_s = 0.0;
    std::uint64_t run_calls = 0;
    double run_s = 0.0;
    std::uint64_t run_ops = 0;
    std::uint64_t wait_calls = 0;
    double wait_s = 0.0;
    std::uint64_t host_readout_calls = 0;
    double host_readout_s = 0.0;
    // Record-path attribution (op capture into TensorGraph).
    std::uint64_t record_get_node_calls = 0;
    double record_get_node_s = 0.0;
    std::uint64_t record_new_nodes = 0;
    std::uint64_t record_pin_calls = 0;
    double record_pin_s = 0.0;
    std::uint64_t record_register_calls = 0;
    double record_register_s = 0.0;
    std::uint64_t record_linear_bwd_calls = 0;
    double record_linear_bwd_s = 0.0;
    std::uint64_t record_ce_bwd_calls = 0;
    double record_ce_bwd_s = 0.0;
    std::uint64_t record_relu_bwd_calls = 0;
    double record_relu_bwd_s = 0.0;
    std::uint64_t record_gemm_calls = 0;
    double record_gemm_s = 0.0;
};

GraphApiTimingStats g_timing;

using SteadyClock = std::chrono::steady_clock;

double seconds_since(SteadyClock::time_point const start)
{
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

void reclaim_pending_outputs_locked()
{
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    if (g_exec->pending_output_reclaim.empty())
    {
        return;
    }
    nntile::Runtime &runtime = *g_exec->runtime;
    // Temps are often del'd after wait(); keep still-marked entries so the
    // next compile/run can invalidate them once NodeRefs drop.
    std::vector<nntile::TensorGraph::TensorNode *> still_marked;
    still_marked.reserve(g_exec->pending_output_reclaim.size());
    for (nntile::TensorGraph::TensorNode *logical :
        g_exec->pending_output_reclaim)
    {
        if (logical == nullptr)
        {
            continue;
        }
        if (logical->is_output() || logical->is_input())
        {
            still_marked.push_back(logical);
            continue;
        }
        runtime.invalidate_logical_tiles(logical);
    }
    g_exec->pending_output_reclaim = std::move(still_marked);
}

void collect_pending_output_reclaim_locked(
    const nntile::TensorGraph::PhaseSnapshot &phase)
{
    if (g_exec == nullptr)
    {
        return;
    }
    // Invalidate unmarked leftovers from a prior phase, but keep entries that
    // are still marked (held across compile) so a later wait() can reclaim.
    reclaim_pending_outputs_locked();
    auto &reclaim = g_exec->pending_output_reclaim;
    std::unordered_set<nntile::TensorGraph::TensorNode *> seen(
        reclaim.begin(),
        reclaim.end());
    reclaim.reserve(reclaim.size() + phase.carried_tensors.size());
    for (nntile::TensorGraph::TensorNode const *t : phase.carried_tensors)
    {
        if (t == nullptr || !t->is_output())
        {
            continue;
        }
        auto *mutable_t = const_cast<nntile::TensorGraph::TensorNode *>(t);
        if (seen.insert(mutable_t).second)
        {
            reclaim.push_back(mutable_t);
        }
    }
}

void sync_param_grad_aliases_locked();

void compile_graph_locked(
    bool clear_pending_after,
    std::vector<at::Tensor> &pin_drop);

void run_graph_locked();

void finish_run_locked();

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

    bool applied_any = false;
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
            applied_any = true;
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
    // Axis tile_sizes changed: drop cached layouts so the next ensure rebuilds
    // from the updated AxisDescriptors.
    if (applied_any && g_exec != nullptr && g_exec->session_tiling != nullptr)
    {
        g_exec->session_tiling->clear();
    }
    // Pending tiling is one-shot: applied at this compile, do not re-apply
    // (and clear session layouts) on every subsequent compile.
    g_axis_tiling_by_name.clear();
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
    nntile::TensorGraph::PhaseSnapshot staging_phase;
    staging_phase.op_begin = g_graph->num_ops();
    staging_phase.op_end = staging_phase.op_begin;
    staging_phase.carried_tensors = {staging};
    if (g_exec->session_tiling == nullptr)
    {
        g_exec->session_tiling =
            std::make_shared<nntile::TensorGraphTiling>();
    }
    g_exec->session_tiling->ensure_phase_layouts(*g_graph, staging_phase);
    nntile::lower_staging_tensor_immediate(
        *g_graph,
        staging,
        g_exec->session_tiling,
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
    if (staging == nullptr)
    {
        return;
    }
    if (count == 0)
    {
        ensure_recorder_exec_state_locked();
        g_exec->runtime->mark_initialized(staging);
        return;
    }
    if (host_ptr == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: staging write requires non-null host pointer");
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
}

void invalidate_staging_tile_submit_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr || g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    nntile::Runtime &runtime = *g_exec->runtime;
    runtime.wait();
    nntile::TileGraph::TileNode *tile =
        require_single_staging_tile_locked(staging);
    switch (staging->dtype())
    {
    case nntile::DataType::FP32:
        runtime.get_tile<nntile::fp32_t>(tile).invalidate_submit();
        break;
    case nntile::DataType::INT64:
        runtime.get_tile<nntile::int64_t>(tile).invalidate_submit();
        break;
    case nntile::DataType::BOOL:
        runtime.get_tile<nntile::bool_t>(tile).invalidate_submit();
        break;
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging invalidate dtype");
    }
    runtime.invalidate_initialized(staging);
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

nntile::TensorGraph::TensorNode *new_ephemeral_staging_node_locked(
    nntile::TensorGraph::TensorNode *logical,
    const std::string &tag)
{
    if (logical == nullptr || g_graph == nullptr)
    {
        return nullptr;
    }
    auto *staging = g_graph->data(logical->shape(), logical->dtype());
    staging->mark_input(false);
    staging->mark_output(false);
    staging->set_name(
        std::string("io_staging_") + logical->name() + "_" + tag + "_" +
        std::to_string(++g_ephemeral_staging_serial));
    return staging;
}

bool should_pin_tensor_for_graph_locked(const at::Tensor &tensor)
{
    return is_metadata_only_tensor(tensor) &&
        nntile_binding(tensor) != nullptr;
}

void pin_tensor_for_graph(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    TensorImplKey const key = tensor_impl_key(tensor);
    for (at::Tensor const &pinned : g_pinned_tensors)
    {
        if (tensor_impl_key(pinned) == key)
        {
            return;
        }
    }
    g_pinned_tensors.push_back(tensor);
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

    if (nntile_binding(mutable_tensor) == nullptr)
    {
        auto new_binding = std::make_shared<NNTileBinding>(node);
        attach_binding(mutable_tensor, new_binding);
    }

    return node;
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

void compact_tensor_graph_session_locked()
{
    // Drop sealed TensorGraph ops so the next record/compile is O(phase).
    // Unsealed ops recorded after the last seal (next phase already in
    // flight while a prior run() completes) are preserved.
    if (g_graph == nullptr)
    {
        return;
    }
    g_graph->drop_all_ops();
    if (g_exec != nullptr)
    {
        g_exec->pending_output_reclaim.clear();
    }
}

void compile_graph_locked(
    bool clear_pending_after,
    std::vector<at::Tensor> &pin_drop)
{
    // Prior async run() must finish before sealing the next phase.
    if (g_run_cleanup_pending)
    {
        finish_run_locked();
    }

    if (g_graph == nullptr ||
        g_graph->num_ops() <= g_graph->phase_seal_cursor())
    {
        return;
    }

    SteadyClock::time_point const t0 = SteadyClock::now();

    ensure_nntile_context();
    ensure_recorder_exec_state_locked();

    sync_param_grad_aliases_locked();
    apply_pending_axis_tiling_locked();

    SteadyClock::time_point t_part = SteadyClock::now();
    const nntile::TensorGraph::PhaseSnapshot phase = g_graph->seal_phase();
    std::vector<nntile::TensorGraph::TensorNode *> scatter_stagings;
    collect_scatter_stagings_from_phase_locked(phase, scatter_stagings);
    collect_pending_output_reclaim_locked(phase);
    g_timing.compile_seal_s += seconds_since(t_part);

    // Phase-scoped tiling: full-graph from_tensor_graph rebuilt layouts for
    // every historical tensor node and made compile O(session length).
    // Session-scoped tiling only constructs layouts for newly touched tensors
    // and is shared into TileGraph (no per-compile deep copy).
    t_part = SteadyClock::now();
    if (g_exec->session_tiling == nullptr)
    {
        g_exec->session_tiling =
            std::make_shared<nntile::TensorGraphTiling>();
    }
    g_exec->session_tiling->ensure_phase_layouts(*g_graph, phase);
    g_timing.compile_tiling_s += seconds_since(t_part);

    t_part = SteadyClock::now();
    try
    {
        nntile::append_tensor_graph_phase(
            *g_graph,
            phase,
            g_exec->session_tiling,
            *g_exec->tile_graph,
            g_exec->inc_state,
            g_exec->tile_map);
    }
    catch (const std::runtime_error &err)
    {
        const std::string msg = err.what();
        if (msg.find("layout_fingerprint mismatch") != std::string::npos)
        {
            throw std::runtime_error(
                msg +
                " Hint: avoid .cpu() / host round-trips on nntile tensors "
                "before the first compile_graph() that applies "
                "set_axis_group_tiling(); early host reads seal the untiled "
                "layout into the TileGraph.");
        }
        throw;
    }
    g_timing.compile_append_s += seconds_since(t_part);

    g_exec->pending_exec_op_begin = g_exec->executed_op_end;
    t_part = SteadyClock::now();
    g_exec->runtime->compile();
    g_timing.compile_runtime_s += seconds_since(t_part);
    g_exec->pending_exec_op_end =
        g_exec->runtime->execution_op_count();
    g_exec->pending_scatter_stagings = std::move(scatter_stagings);

    if (clear_pending_after)
    {
        transfer_pinned_tensors_locked(pin_drop);
        g_defer_pending_clear_after_run = true;
    }

    std::uint64_t const phase_ops = static_cast<std::uint64_t>(
        g_exec->pending_exec_op_end - g_exec->pending_exec_op_begin);
    g_timing.compile_s += seconds_since(t0);
    ++g_timing.compile_calls;
    g_timing.compile_ops += phase_ops;
}

void run_graph_locked()
{
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    // Submit only. Never wait here — callers use wait_graph_session() /
    // torch_nntile.wait() for synchronization and post-run reclaim.
    if (g_exec->pending_exec_op_end > g_exec->pending_exec_op_begin)
    {
        std::uint64_t const phase_ops = static_cast<std::uint64_t>(
            g_exec->pending_exec_op_end - g_exec->pending_exec_op_begin);
        SteadyClock::time_point const t0 = SteadyClock::now();
        g_exec->runtime->execute_range(
            g_exec->pending_exec_op_begin,
            g_exec->pending_exec_op_end);
        g_timing.run_s += seconds_since(t0);
        ++g_timing.run_calls;
        g_timing.run_ops += phase_ops;
        g_exec->executed_op_end = g_exec->pending_exec_op_end;
        g_exec->pending_exec_op_begin = g_exec->pending_exec_op_end;
    }
    // Always require wait() for compile-side cleanup (pin_hold, reclaim,
    // scatter stagings, deferred pending clear) — including empty post-DCE
    // phases that submit no StarPU tasks.
    g_run_cleanup_pending = true;
}

void finish_run_locked()
{
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        g_run_cleanup_pending = false;
        return;
    }
    // run() is asynchronous: only wait when a submit is still unfinished.
    // Idle calls (e.g. redundant wait_for_all before .to("cpu")) are no-ops.
    if (!g_run_cleanup_pending)
    {
        return;
    }
    SteadyClock::time_point const t0 = SteadyClock::now();
    g_exec->runtime->wait();
    for (nntile::TensorGraph::TensorNode *staging :
        g_exec->pending_scatter_stagings)
    {
        invalidate_staging_tile_submit_locked(staging);
        // .to("nntile") marks ingress staging as input; clear marks after
        // scatter completes so later seals do not keep carrying stagings.
        // Drop incremental tile state so the next compile cannot reuse the
        // invalidated staging tile nodes and allocate empty replacements.
        staging->mark_input(false);
        staging->mark_output(false);
        g_exec->inc_state.tensor_to_tiles.erase(staging);
        g_exec->inc_state.tensor_layout_fp.erase(staging);
        g_exec->tile_map.erase(staging);
        if (g_exec->session_tiling != nullptr)
        {
            g_exec->session_tiling->erase(staging);
        }
    }
    g_exec->pending_scatter_stagings.clear();
    if (g_defer_pending_clear_after_run)
    {
        std::vector<at::Tensor> pin_drop;
        clear_pending_graph_after_compile_locked(pin_drop);
        g_defer_pending_clear_after_run = false;
    }
    // Drop compile-time pin_hold inside the locked section so mark_output
    // flips are visible before reclaim.
    g_exec->pin_hold.clear();
    reclaim_pending_outputs_locked();
    // Compact TensorGraph history so the next record/compile is O(phase).
    compact_tensor_graph_session_locked();
    g_run_cleanup_pending = false;
    g_timing.wait_s += seconds_since(t0);
    ++g_timing.wait_calls;
}

void reset_recorder_locked(
    bool clear_tensor_gc,
    std::vector<at::Tensor> &pin_drop)
{
    // Destroy tensors that hold NodeRefs while TensorGraph nodes are still
    // alive so ~NNTileBinding can safely call mark_output(false).
    if (g_exec != nullptr && g_exec->runtime != nullptr)
    {
        g_exec->runtime->wait();
    }
    g_run_cleanup_pending = false;
    if (g_exec != nullptr)
    {
        g_exec->pin_hold.clear();
    }
    {
        std::vector<at::Tensor> pins;
        transfer_pinned_tensors_locked(pins);
        // pins destroyed here, before g_graph.reset().
    }
    clear_param_grad_registry_locked();
    g_relu_preactivation_stack.clear();
    g_defer_pending_clear_after_run = false;
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
    g_ephemeral_staging_serial = 0;
    g_exec.reset();
    set_logical_tensor_nodes_alive(false);
    g_graph.reset();
    drain_starpu_after_session_teardown();
    if (clear_tensor_gc)
    {
        clear_tensor_gc_state();
    }
    // Callers historically destroyed pin_drop after unlock; pins are now
    // released above while the graph is alive.
    pin_drop.clear();
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
        // Autograd only populates .grad on leaves. Linear/transpose backward
        // also register activation tensors here; accessing .grad on those
        // non-leaves triggers TensorBody warnings (pytorch#30531).
        if (!entry.param.is_leaf())
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
    // compile + run only. Never wait here — callers must use wait() /
    // wait_graph_session() (same contract as compile_graph + run).
    compile_graph_locked(true, pin_drop);
    run_graph_locked();
}

void shutdown_recorder_locked(std::vector<at::Tensor> &pin_drop)
{
    g_defer_pending_clear_after_run = false;
    reset_recorder_locked(true, pin_drop);
}

} // namespace

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
                        "torch_nntile.compile_graph()/run() or execute() "
                        "before ") +
            op_name);
    }
}

void execute_pending_graph()
{
    std::vector<at::Tensor> pin_drop;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        execute_pending_graph_locked(pin_drop);
        if (g_exec != nullptr && !pin_drop.empty())
        {
            g_exec->pin_hold = std::move(pin_drop);
        }
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
}

void wait_graph_session()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    finish_run_locked();
    if (starpu_is_initialized())
    {
        starpu_task_wait_for_all();
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
    void *host_ptr,
    nntile::DataType dtype,
    std::size_t count)
{
    if (logical == nullptr || host_ptr == nullptr || count == 0)
    {
        return;
    }
    if (g_graph == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: gather readout requires an active TensorGraph");
    }
    SteadyClock::time_point const t0 = SteadyClock::now();
    // Finish a prior async run() before recording gather ops. Otherwise
    // compile_graph_locked() would wait+compact and drop_all_ops() would
    // wipe the clear/gather we are about to append.
    if (g_run_cleanup_pending)
    {
        finish_run_locked();
    }
    nntile::TensorGraph::TensorNode *staging =
        new_ephemeral_staging_node_locked(logical, "readout");
    if (staging == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: gather readout failed to create staging tensor");
    }
    // Output S: single-tile only; lowered during compile after gather is recorded.
    staging->mark_output(true);
    nntile::tensor::clear(staging);
    nntile::tensor::gather(logical, staging);

    std::vector<at::Tensor> pin_drop;
    compile_graph_locked(false, pin_drop);
    run_graph_locked();
    finish_run_locked();

    read_staging_to_host_locked(staging, host_ptr, dtype, count);
    invalidate_staging_tile_submit_locked(staging);
    staging->mark_output(false);
    g_timing.host_readout_s += seconds_since(t0);
    ++g_timing.host_readout_calls;
}

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    NodeRef binding = nntile_binding(src);
    if (binding == nullptr || binding->logical == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: copy nntile tensor to CPU requires a bound "
            "logical graph node (use .to('nntile') first)");
    }
    nntile::TensorGraph::TensorNode *logical = binding->logical;
    const nntile::DataType dtype = logical->dtype();
    const std::size_t count =
        static_cast<std::size_t>(logical->nelems());
    void *host_ptr = dst.storage().data_ptr().get();

    // Sync a prior async execute()/run() even when no ops are pending so
    // subsequent gather recording is not wiped by wait-side drop_all_ops().
    if (g_run_cleanup_pending)
    {
        finish_run_locked();
    }

    if (g_graph != nullptr &&
        g_graph->num_ops() > g_graph->phase_seal_cursor())
    {
        std::vector<at::Tensor> pin_drop;
        compile_graph_locked(false, pin_drop);
        run_graph_locked();
        finish_run_locked();
    }

    if (g_exec == nullptr || g_exec->runtime == nullptr ||
        !g_exec->runtime->is_compiled())
    {
        throw std::runtime_error(
            "torch_nntile: copy nntile tensor to CPU requires compile_graph() "
            "and run() first");
    }

    gather_logical_to_staging_and_read_locked(
        logical,
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
        set_logical_tensor_nodes_alive(true);
    }

    const std::vector<nntile::Index> shape =
        aten_sizes_to_graph_shape(cpu_src.sizes());
    const nntile::DataType dtype =
        aten_scalar_to_nntile_dtype(cpu_src.scalar_type());
    const TensorImplKey impl_key = tensor_impl_key(nntile_dst);

    if (NodeRef existing = nntile_binding(nntile_dst);
        existing != nullptr && existing->logical != nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: CPU→nntile copy into an already-bound tensor is "
            "unsupported; ingress each tensor once via .to('nntile')");
    }

    auto *logical = g_graph->data(shape, dtype);
    apply_axis_name_hints_locked(impl_key, logical);
    // Host-ingressed tensors are persistent inputs for the session.
    logical->mark_input(true);

    auto binding = std::make_shared<NNTileBinding>(logical);
    attach_binding(nntile_dst, binding);

    auto *staging = new_ephemeral_staging_node_locked(logical, "ingress");
    if (staging == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: failed to create ingress staging tensor");
    }
    staging->mark_input(true);
    lower_io_staging_locked(staging);
    write_cpu_bytes_to_staging_locked(
        staging,
        cpu_src.storage().data_ptr().get(),
        dtype,
        static_cast<std::size_t>(cpu_src.numel()));

    nntile::tensor::scatter(staging, logical);

    g_pinned_tensors.push_back(nntile_dst);
}


nntile::TensorGraph::TensorNode *get_or_create_data_node(
    const at::Tensor &tensor,
    const std::vector<nntile::Index> &shape,
    nntile::DataType dtype,
    bool mark_as_input)
{
    const SteadyClock::time_point t0 = SteadyClock::now();
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
        set_logical_tensor_nodes_alive(true);
    }

    const std::size_t data_before = g_graph->num_data();
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    at::Tensor mutable_tensor = const_cast<at::Tensor &>(tensor);
    nntile::TensorGraph::TensorNode *node = logical_node_for_tensor_locked(
        mutable_tensor,
        impl_key,
        shape,
        dtype,
        mark_as_input);
    assert_has_node_ref(tensor, "get_or_create_data_node");
    if (g_graph->num_data() > data_before)
    {
        g_timing.record_new_nodes +=
            static_cast<std::uint64_t>(g_graph->num_data() - data_before);
    }
    ++g_timing.record_get_node_calls;
    g_timing.record_get_node_s += seconds_since(t0);
    return node;
}

void register_data_node(
    const at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *node)
{
    const SteadyClock::time_point t0 = SteadyClock::now();
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    at::Tensor mutable_tensor = tensor;
    if (nntile_binding(mutable_tensor) == nullptr)
    {
        attach_binding(
            mutable_tensor,
            std::make_shared<NNTileBinding>(node));
    }
    assert_has_node_ref(tensor, "register_data_node");
    ++g_timing.record_register_calls;
    g_timing.record_register_s += seconds_since(t0);
}

void note_record_linear_bwd(double seconds)
{
    ++g_timing.record_linear_bwd_calls;
    g_timing.record_linear_bwd_s += seconds;
}

void note_record_ce_bwd(double seconds)
{
    ++g_timing.record_ce_bwd_calls;
    g_timing.record_ce_bwd_s += seconds;
}

void note_record_relu_bwd(double seconds)
{
    ++g_timing.record_relu_bwd_calls;
    g_timing.record_relu_bwd_s += seconds;
}

void note_record_gemm(double seconds)
{
    ++g_timing.record_gemm_calls;
    g_timing.record_gemm_s += seconds;
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
    g_param_grad_registry[key] = ParamGradEntry{grad_node, param};
}

nntile::TensorGraph::TensorNode *lookup_param_grad_node(
    const at::Tensor &param)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const auto found = g_param_grad_registry.find(tensor_impl_key(param));
    if (found == g_param_grad_registry.end())
    {
        return nullptr;
    }
    return found->second.grad_node;
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
    g_param_grad_registry.erase(key);
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

void pin_graph_op_inputs(const std::vector<at::Tensor> &inputs)
{
    const SteadyClock::time_point t0 = SteadyClock::now();
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    for (const at::Tensor &tensor : inputs)
    {
        if (should_pin_tensor_for_graph_locked(tensor))
        {
            pin_tensor_for_graph(tensor);
        }
    }
    ++g_timing.record_pin_calls;
    g_timing.record_pin_s += seconds_since(t0);
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
    const at::Tensor &tensor,
    const std::unordered_map<int, std::string> &names)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const int ndim = static_cast<int>(tensor.dim());
    nntile::TensorGraph::TensorNode *bound_node = nntile_node(tensor);

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

        nntile::TensorGraph::TensorNode *node = bound_node;
        if (node == nullptr)
        {
            node = node_for_impl_locked(impl_key);
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

std::string format_info_locked()
{
    auto avg_ms = [](double seconds, std::uint64_t calls) -> double
    {
        if (calls == 0)
        {
            return 0.0;
        }
        return 1.0e3 * seconds / static_cast<double>(calls);
    };

    std::ostringstream ss;
    ss << "torch_nntile graph API timing (cumulative):\n";
    ss << "  compile_graph: " << g_timing.compile_calls << " calls, "
       << g_timing.compile_s << "s"
       << " (avg " << avg_ms(g_timing.compile_s, g_timing.compile_calls)
       << " ms), tile-ops lowered=" << g_timing.compile_ops << '\n';
    if (g_timing.compile_calls > 0)
    {
        ss << "    seal+reclaim: " << g_timing.compile_seal_s << "s"
           << " (avg "
           << avg_ms(g_timing.compile_seal_s, g_timing.compile_calls)
           << " ms)\n";
        ss << "    from_phase:   " << g_timing.compile_tiling_s << "s"
           << " (avg "
           << avg_ms(g_timing.compile_tiling_s, g_timing.compile_calls)
           << " ms)\n";
        ss << "    append_phase: " << g_timing.compile_append_s << "s"
           << " (avg "
           << avg_ms(g_timing.compile_append_s, g_timing.compile_calls)
           << " ms)\n";
        ss << "    runtime.compile: " << g_timing.compile_runtime_s << "s"
           << " (avg "
           << avg_ms(g_timing.compile_runtime_s, g_timing.compile_calls)
           << " ms)\n";
    }
    ss << "  run (submit):  " << g_timing.run_calls << " calls, "
       << g_timing.run_s << "s"
       << " (avg " << avg_ms(g_timing.run_s, g_timing.run_calls)
       << " ms), tile-ops submitted=" << g_timing.run_ops << '\n';
    ss << "  wait:          " << g_timing.wait_calls << " calls, "
       << g_timing.wait_s << "s"
       << " (avg " << avg_ms(g_timing.wait_s, g_timing.wait_calls)
       << " ms; finishes a pending run() only)\n";
    ss << "  host_readout:  " << g_timing.host_readout_calls << " calls, "
       << g_timing.host_readout_s << "s"
       << " (avg "
       << avg_ms(g_timing.host_readout_s, g_timing.host_readout_calls)
       << " ms; .to(\"cpu\") gather wall, includes nested "
       << "compile/run/wait)\n";
    ss << "  sum compile+run+wait: "
       << (g_timing.compile_s + g_timing.run_s + g_timing.wait_s) << "s\n";
    if (g_timing.record_get_node_calls > 0 ||
        g_timing.record_linear_bwd_calls > 0)
    {
        ss << "  record path (TensorGraph capture):\n";
        ss << "    get_or_create_node: " << g_timing.record_get_node_calls
           << " calls, " << g_timing.record_get_node_s << "s (avg "
           << avg_ms(
                  g_timing.record_get_node_s,
                  g_timing.record_get_node_calls)
           << " ms), new_nodes=" << g_timing.record_new_nodes << '\n';
        ss << "    pin_inputs: " << g_timing.record_pin_calls
           << " calls, " << g_timing.record_pin_s << "s (avg "
           << avg_ms(g_timing.record_pin_s, g_timing.record_pin_calls)
           << " ms)\n";
        ss << "    register_data_node: " << g_timing.record_register_calls
           << " calls, " << g_timing.record_register_s << "s\n";
        ss << "    gemm record: " << g_timing.record_gemm_calls
           << " calls, " << g_timing.record_gemm_s << "s (avg "
           << avg_ms(g_timing.record_gemm_s, g_timing.record_gemm_calls)
           << " ms)\n";
        ss << "    linear_backward: " << g_timing.record_linear_bwd_calls
           << " calls, " << g_timing.record_linear_bwd_s << "s (avg "
           << avg_ms(
                  g_timing.record_linear_bwd_s,
                  g_timing.record_linear_bwd_calls)
           << " ms)\n";
        ss << "    ce_backward: " << g_timing.record_ce_bwd_calls
           << " calls, " << g_timing.record_ce_bwd_s << "s (avg "
           << avg_ms(
                  g_timing.record_ce_bwd_s, g_timing.record_ce_bwd_calls)
           << " ms)\n";
        ss << "    relu/threshold_backward: "
           << g_timing.record_relu_bwd_calls << " calls, "
           << g_timing.record_relu_bwd_s << "s (avg "
           << avg_ms(
                  g_timing.record_relu_bwd_s,
                  g_timing.record_relu_bwd_calls)
           << " ms)\n";
    }
    if (g_timing.run_calls > 0)
    {
        ss << "  note: wait_calls should be ≈ run_calls when callers avoid "
           << "redundant wait(); idle wait() is a no-op\n";
    }

    if (g_graph != nullptr)
    {
        ss << "  session tensor_graph_ops: " << g_graph->num_ops()
           << " (seal_cursor=" << g_graph->phase_seal_cursor() << ")\n";
    }
    if (g_exec != nullptr && g_exec->runtime != nullptr)
    {
        ss << "  session executed_tile_ops: " << g_exec->executed_op_end
           << " / " << g_exec->runtime->execution_op_count() << '\n';
    }
    return ss.str();
}

void print_info()
{
    std::string text;
    {
        std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
        text = format_info_locked();
    }
    std::fputs(text.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace torch_nntile

#else

#include <cstring>

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

void wait_graph_session()
{
}

void reset_graph_session()
{
    require_libnntile();
}

bool has_graph_session()
{
    return false;
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
    const at::Tensor & /*tensor*/,
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

void print_info()
{
}

void shutdown_recorder()
{
}

void copy_nntile_tensor_to_cpu(const at::Tensor & /*src*/, at::Tensor & /*dst*/)
{
}

} // namespace torch_nntile

#endif
