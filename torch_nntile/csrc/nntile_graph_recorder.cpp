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
std::unordered_map<TensorImplKey, std::vector<std::int64_t>> g_label_host_cache;
std::unordered_map<TensorImplKey, std::unordered_map<int, std::string>>
    g_axis_name_hints;
std::unordered_map<std::string, std::vector<nntile::Index>> g_axis_tiling_by_name;

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

void write_mapped_tensor_locked(
    TensorImplKey impl_key,
    const MappedTensor &mapped)
{
    g_tensor_nodes[impl_key] = mapped;
}

void ensure_meta_from_mapped_locked(
    const at::Tensor &tensor,
    const MappedTensor &mapped)
{
    if (mapped.node == nullptr || nntile_binding(tensor) != nullptr)
    {
        return;
    }
    auto binding = std::make_shared<NNTileBinding>(mapped.node);
    binding->io_staging = mapped.staging_node;
    at::Tensor mutable_tensor = tensor;
    attach_binding(mutable_tensor, binding);
}

void attach_node_binding(
    at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *node,
    const MappedTensor &mapped)
{
    if (node == nullptr || nntile_binding(tensor) != nullptr)
    {
        return;
    }
    auto binding = std::make_shared<NNTileBinding>(node);
    binding->io_staging = mapped.staging_node;
    attach_binding(tensor, binding);
}

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

bool should_bind_mapped_at_compile(
    TensorImplKey impl_key,
    const MappedTensor &mapped)
{
    return mapped.bind_at_execute || mapped.is_persistent_input ||
        is_staged_input_impl(impl_key);
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

void bind_host_bytes_to_staging_locked(
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
    switch (dtype)
    {
    case nntile::DataType::FP32:
        runtime.bind_data(
            staging,
            static_cast<const float *>(host_ptr),
            count);
        break;
    case nntile::DataType::INT64:
        runtime.bind_data(
            staging,
            static_cast<const std::int64_t *>(host_ptr),
            count);
        break;
    case nntile::DataType::BOOL:
        runtime.bind_data(
            staging,
            reinterpret_cast<const bool *>(host_ptr),
            count);
        break;
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging bind dtype");
    }
}

void invalidate_staging_tile_buffer_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr || g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    const auto found = g_exec->tile_map.find(staging);
    if (found == g_exec->tile_map.end() || found->second.size() != 1)
    {
        return;
    }
    nntile::TileGraph::TileNode *tile = found->second[0];
    switch (staging->dtype())
    {
    case nntile::DataType::FP32:
        g_exec->runtime->get_tile<nntile::fp32_t>(tile).invalidate_submit();
        break;
    case nntile::DataType::INT64:
        g_exec->runtime->get_tile<nntile::int64_t>(tile).invalidate_submit();
        break;
    case nntile::DataType::BOOL:
        g_exec->runtime->get_tile<nntile::bool_t>(tile).invalidate_submit();
        break;
    default:
        break;
    }
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

    const auto tile_it = g_exec->tile_map.find(staging);
    if (tile_it != g_exec->tile_map.end() && tile_it->second.size() == 1)
    {
        nntile::TileGraph::TileNode *tile = tile_it->second[0];
        if (tile != nullptr && (tile->is_input() || tile->is_output()))
        {
            switch (dtype)
            {
            case nntile::DataType::FP32:
            {
                const std::vector<float> out = runtime.get_output<float>(tile);
                if (out.size() != count)
                {
                    throw std::runtime_error(
                        "torch_nntile: staging tile read size mismatch");
                }
                std::memcpy(host_ptr, out.data(), count * sizeof(float));
                return;
            }
            case nntile::DataType::INT64:
            {
                const std::vector<std::int64_t> out =
                    runtime.get_output<std::int64_t>(tile);
                if (out.size() != count)
                {
                    throw std::runtime_error(
                        "torch_nntile: staging tile read size mismatch");
                }
                std::memcpy(
                    host_ptr,
                    out.data(),
                    count * sizeof(std::int64_t));
                return;
            }
            case nntile::DataType::BOOL:
            {
                const std::vector<bool> out = runtime.get_output<bool>(tile);
                if (out.size() != count)
                {
                    throw std::runtime_error(
                        "torch_nntile: staging tile read size mismatch");
                }
                for (std::size_t i = 0; i < count; ++i)
                {
                    static_cast<bool *>(host_ptr)[i] = out[i];
                }
                return;
            }
            default:
                break;
            }
        }
    }

    switch (dtype)
    {
    case nntile::DataType::FP32:
    {
        const std::vector<float> out = runtime.get_output<float>(staging);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        std::memcpy(host_ptr, out.data(), count * sizeof(float));
        break;
    }
    case nntile::DataType::INT64:
    {
        const std::vector<std::int64_t> out =
            runtime.get_output<std::int64_t>(staging);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        std::memcpy(
            host_ptr,
            out.data(),
            count * sizeof(std::int64_t));
        break;
    }
    case nntile::DataType::BOOL:
    {
        const std::vector<bool> out = runtime.get_output<bool>(staging);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: staging read size mismatch");
        }
        for (std::size_t i = 0; i < count; ++i)
        {
            static_cast<bool *>(host_ptr)[i] = out[i];
        }
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging read dtype");
    }
}

void read_logical_to_host_locked(
    nntile::TensorGraph::TensorNode *logical,
    void *host_ptr,
    nntile::DataType dtype,
    std::size_t count);

bool staging_ready_for_direct_read_locked(
    nntile::TensorGraph::TensorNode *staging);

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

void populate_staging_from_logical_locked(
    nntile::TensorGraph::TensorNode *staging,
    nntile::TensorGraph::TensorNode *source_logical,
    nntile::DataType dtype,
    std::size_t count)
{
    if (staging == nullptr || source_logical == nullptr || count == 0)
    {
        return;
    }
    if (g_exec == nullptr || g_exec->runtime == nullptr ||
        !g_exec->runtime->is_initialized(source_logical))
    {
        return;
    }
    switch (dtype)
    {
    case nntile::DataType::FP32:
    {
        std::vector<float> host(count);
        read_logical_to_host_locked(
            source_logical,
            host.data(),
            dtype,
            count);
        bind_host_bytes_to_staging_locked(
            staging,
            host.data(),
            dtype,
            count);
        break;
    }
    case nntile::DataType::INT64:
    {
        std::vector<std::int64_t> host(count);
        read_logical_to_host_locked(
            source_logical,
            host.data(),
            dtype,
            count);
        bind_host_bytes_to_staging_locked(
            staging,
            host.data(),
            dtype,
            count);
        break;
    }
    case nntile::DataType::BOOL:
    {
        std::vector<unsigned char> host(count);
        read_logical_to_host_locked(
            source_logical,
            host.data(),
            dtype,
            count);
        bind_host_bytes_to_staging_locked(
            staging,
            host.data(),
            dtype,
            count);
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging populate dtype");
    }
}

void refresh_input_scatter_locked(
    at::Tensor &tensor,
    nntile::TensorGraph::TensorNode *logical,
    nntile::TensorGraph::TensorNode *prev_logical,
    nntile::TensorGraph::TensorNode *prev_staging)
{
    if (logical == nullptr || g_graph == nullptr)
    {
        return;
    }
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr)
    {
        return;
    }
    auto *staging = ensure_io_staging_node_locked(binding);
    if (staging == nullptr)
    {
        return;
    }
    if (g_exec != nullptr && g_exec->runtime != nullptr &&
        !g_exec->runtime->is_initialized(staging))
    {
        if (prev_logical != nullptr && prev_logical != logical)
        {
            populate_staging_from_logical_locked(
                staging,
                prev_logical,
                logical->dtype(),
                static_cast<std::size_t>(logical->nelems()));
        }
        else if (
            prev_staging != nullptr &&
            staging_ready_for_direct_read_locked(prev_staging))
        {
            const std::size_t count =
                static_cast<std::size_t>(logical->nelems());
            switch (logical->dtype())
            {
            case nntile::DataType::FP32:
            {
                std::vector<float> host(count);
                read_staging_to_host_locked(
                    prev_staging,
                    host.data(),
                    logical->dtype(),
                    count);
                bind_host_bytes_to_staging_locked(
                    staging,
                    host.data(),
                    logical->dtype(),
                    count);
                break;
            }
            case nntile::DataType::INT64:
            {
                std::vector<std::int64_t> host(count);
                read_staging_to_host_locked(
                    prev_staging,
                    host.data(),
                    logical->dtype(),
                    count);
                bind_host_bytes_to_staging_locked(
                    staging,
                    host.data(),
                    logical->dtype(),
                    count);
                break;
            }
            case nntile::DataType::BOOL:
            {
                std::vector<unsigned char> host(count);
                read_staging_to_host_locked(
                    prev_staging,
                    host.data(),
                    logical->dtype(),
                    count);
                bind_host_bytes_to_staging_locked(
                    staging,
                    host.data(),
                    logical->dtype(),
                    count);
                break;
            }
            default:
                break;
            }
        }
    }
    nntile::tensor::scatter(staging, logical);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end())
    {
        found->second.staging_node = staging;
    }
}

bool should_pin_tensor_for_graph_locked(const at::Tensor &tensor)
{
    if (has_host_staging(tensor))
    {
        return true;
    }
    if (is_staged_input_tensor(tensor))
    {
        return true;
    }
    if (!is_metadata_only_tensor(tensor))
    {
        return false;
    }
    const auto found = g_tensor_nodes.find(tensor_impl_key(tensor));
    return found != g_tensor_nodes.end() && found->second.is_persistent_input;
}

bool staging_ready_for_direct_read_locked(
    nntile::TensorGraph::TensorNode *staging)
{
    return staging != nullptr && g_exec != nullptr &&
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

void read_logical_to_host_locked(
    nntile::TensorGraph::TensorNode *logical,
    void *host_ptr,
    nntile::DataType dtype,
    std::size_t count)
{
    if (logical == nullptr || host_ptr == nullptr || count == 0)
    {
        return;
    }
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: no runtime for logical readout");
    }
    nntile::Runtime &runtime = *g_exec->runtime;
    runtime.wait();
    switch (dtype)
    {
    case nntile::DataType::FP32:
    {
        const std::vector<float> out = runtime.get_output<float>(logical);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: logical read size mismatch");
        }
        std::memcpy(host_ptr, out.data(), count * sizeof(float));
        break;
    }
    case nntile::DataType::INT64:
    {
        const std::vector<std::int64_t> out =
            runtime.get_output<std::int64_t>(logical);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: logical read size mismatch");
        }
        std::memcpy(
            host_ptr,
            out.data(),
            count * sizeof(std::int64_t));
        break;
    }
    case nntile::DataType::BOOL:
    {
        const std::vector<bool> out = runtime.get_output<bool>(logical);
        if (out.size() != count)
        {
            throw std::runtime_error(
                "torch_nntile: logical read size mismatch");
        }
        for (std::size_t i = 0; i < count; ++i)
        {
            static_cast<bool *>(host_ptr)[i] = out[i];
        }
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported logical read dtype");
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
        MappedTensor mapped{};
        const auto found = g_tensor_nodes.find(impl_key);
        if (found != g_tensor_nodes.end())
        {
            mapped = found->second;
        }
        mapped.node = node;
        mapped.staging_node = binding->io_staging;
        mapped.dtype = dtype;
        mapped.count = static_cast<std::size_t>(graph_numel(shape));
        write_mapped_tensor_locked(impl_key, mapped);
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

    MappedTensor mapped{
        node,
        nullptr,
        dtype,
        static_cast<std::size_t>(graph_numel(shape)),
        false,
        false,
        false,
        nullptr};
    write_mapped_tensor_locked(impl_key, mapped);
    return node;
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
    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end() && found->second.node != nullptr)
    {
        return found->second.node;
    }
    for (const at::Tensor &tensor : g_pinned_tensors)
    {
        if (tensor_impl_key(tensor) != impl_key)
        {
            continue;
        }
        if (NodeRef binding = nntile_binding(tensor); binding != nullptr)
        {
            return binding->logical;
        }
    }
    return nullptr;
}

nntile::TensorGraph::TensorNode *session_node_for_tensor_locked(
    const at::Tensor &tensor)
{
    const TensorImplKey impl_key = canonical_tensor_impl_key(tensor);
    return session_node_for_impl_locked(impl_key);
}

void copy_host_visible_outputs(
    nntile::Runtime &runtime,
    const void * /*preferred_ptr*/)
{
    for (auto &[impl_key, mapped] : g_tensor_nodes)
    {
        if (!mapped.needs_host_copy || mapped.host_data_ptr == nullptr)
        {
            continue;
        }
        nntile::TensorGraph::TensorNode *node = mapped.node;
        if (node == nullptr)
        {
            node = session_node_for_impl_locked(impl_key);
        }
        if (node == nullptr)
        {
            continue;
        }
        MappedTensor copy_mapped = mapped;
        copy_mapped.node = node;
        if (node->is_output())
        {
            copy_output_if_needed(runtime, copy_mapped, mapped.host_data_ptr);
        }
        else
        {
            copy_tensor_from_runtime(
                runtime,
                copy_mapped,
                mapped.host_data_ptr);
        }
    }
}

void sync_current_run_visible_outputs_locked()
{
    (void)0;
    // Phase 7: outputs are read via gather + staging, not host Storage.
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
        for (nntile::TensorGraph::TensorNode *staging :
            g_exec->pending_scatter_stagings)
        {
            invalidate_staging_tile_buffer_locked(staging);
        }
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
    g_tensor_nodes.clear();
    g_param_grad_nodes.clear();
    clear_param_grad_registry_locked();
    g_relu_preactivation_stack.clear();
    g_all_nodes.clear();
    transfer_pinned_tensors_locked(pin_drop);
    g_label_host_cache.clear();
    g_defer_pending_clear_after_run = false;
    g_axis_name_hints.clear();
    g_axis_tiling_by_name.clear();
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
    if (!is_metadata_only_tensor(grad) && !has_host_staging(grad))
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
        host_ptr != nullptr && !is_metadata_only_tensor(grad),
        false,
        is_staged_input_tensor(grad) && !is_metadata_only_tensor(grad),
        host_ptr};
    attach_node_binding(grad, grad_node, g_tensor_nodes[tensor_impl_key(grad)]);
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

bool read_nntile_logical_to_host(const at::Tensor &tensor, void *host_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (host_ptr == nullptr)
    {
        return false;
    }
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr || binding->logical == nullptr)
    {
        return false;
    }
    if (g_exec == nullptr || g_exec->runtime == nullptr ||
        !g_exec->runtime->is_compiled() ||
        !g_exec->runtime->is_initialized(binding->logical))
    {
        return false;
    }
    const nntile::DataType dtype = binding->logical->dtype();
    const std::size_t count =
        static_cast<std::size_t>(binding->logical->nelems());
    try
    {
        read_logical_to_host_locked(
            binding->logical,
            host_ptr,
            dtype,
            count);
    }
    catch (const std::exception &)
    {
        return false;
    }
    return true;
}

const std::int64_t *label_host_cache_ptr(
    const at::Tensor &tensor,
    std::size_t *out_count)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_label_host_cache.find(impl_key);
    if (found == g_label_host_cache.end() || found->second.empty())
    {
        return nullptr;
    }
    if (out_count != nullptr)
    {
        *out_count = found->second.size();
    }
    return found->second.data();
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
    return session_node_for_impl_locked(impl_key);
}

void sync_nntile_storage_to_runtime(void *host_data_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_exec == nullptr || g_exec->runtime == nullptr)
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
    bind_storage_to_runtime(*g_exec->runtime, host_data_ptr, mapped);
}

void sync_runtime_to_nntile_storage(void *host_data_ptr)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_exec == nullptr || g_exec->runtime == nullptr)
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
    g_exec->runtime->wait();
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
    copy_tensor_from_runtime(*g_exec->runtime, mapped, host_data_ptr);
}

void sync_runtime_to_nntile_tensor(const at::Tensor &tensor)
{
    if (!has_host_staging(tensor))
    {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    const TensorImplKey impl_key = canonical_tensor_impl_key(tensor);
    nntile::TensorGraph::TensorNode *node =
        session_node_for_impl_locked(impl_key);
    if (node == nullptr)
    {
        return;
    }
    g_exec->runtime->wait();
    void *host_data_ptr = tensor.storage().data_ptr().get();
    MappedTensor mapped;
    mapped.node = node;
    mapped.dtype = node->dtype();
    mapped.count = static_cast<std::size_t>(node->nelems());
    mapped.needs_host_copy = true;
    mapped.host_data_ptr = host_data_ptr;
    copy_tensor_from_runtime(*g_exec->runtime, mapped, host_data_ptr);
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
    nntile::tensor::gather(logical, staging);

    std::vector<at::Tensor> pin_drop;
    compile_graph_locked(false, pin_drop);
    run_graph_locked();

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
        bind_host_bytes_to_staging_locked(
            staging,
            cpu_src.storage().data_ptr().get(),
            dtype,
            static_cast<std::size_t>(cpu_src.numel()));
        if (dtype == nntile::DataType::INT64)
        {
            const std::size_t count =
                static_cast<std::size_t>(cpu_src.numel());
            g_label_host_cache[impl_key].resize(count);
            std::memcpy(
                g_label_host_cache[impl_key].data(),
                cpu_src.storage().data_ptr().get(),
                count * sizeof(std::int64_t));
        }
        nntile::tensor::scatter(staging, logical);
        MappedTensor mapped{
            logical,
            staging,
            dtype,
            static_cast<std::size_t>(graph_numel(shape)),
            false,
            false,
            true,
            nullptr};
        write_mapped_tensor_locked(impl_key, mapped);
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
    mark_metadata_only_tensor(nntile_dst);

    lower_io_staging_locked(staging);
    bind_host_bytes_to_staging_locked(
        staging,
        cpu_src.storage().data_ptr().get(),
        dtype,
        static_cast<std::size_t>(cpu_src.numel()));

    if (dtype == nntile::DataType::INT64)
    {
        const std::size_t count =
            static_cast<std::size_t>(cpu_src.numel());
        g_label_host_cache[impl_key].resize(count);
        std::memcpy(
            g_label_host_cache[impl_key].data(),
            cpu_src.storage().data_ptr().get(),
            count * sizeof(std::int64_t));
    }

    nntile::tensor::scatter(staging, logical);

    MappedTensor mapped{
        logical,
        staging,
        dtype,
        static_cast<std::size_t>(graph_numel(shape)),
        false,
        false,
        true,
        nullptr};
    write_mapped_tensor_locked(impl_key, mapped);
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

    const TensorImplKey impl_key = tensor_impl_key(tensor);
    if (tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.storage().nbytes() == 0 &&
        !is_metadata_only_tensor(tensor))
    {
        at::Tensor mutable_tensor = tensor;
        mark_metadata_only_tensor(mutable_tensor);
    }
    void *host_ptr = nullptr;
    if (has_host_staging(tensor))
    {
        host_ptr = tensor.storage().data_ptr().get();
    }
    const bool staged = is_staged_input_tensor(tensor);
    const bool needs_host = has_host_staging(tensor) &&
        (staged || node->is_output());

    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end() && found->second.is_persistent_input)
    {
        found->second.node = node;
        found->second.dtype = node->dtype();
        found->second.count = static_cast<std::size_t>(node->nelems());
        found->second.needs_host_copy = true;
        found->second.host_data_ptr = host_ptr;
        at::Tensor mutable_tensor = tensor;
        if (nntile_binding(mutable_tensor) == nullptr)
        {
            attach_node_binding(mutable_tensor, node, found->second);
        }
        assert_has_node_ref(tensor, "register_data_node");
        return;
    }

    bool bind_at_execute = staged;
    bool is_persistent = staged;
    if (found != g_tensor_nodes.end())
    {
        bind_at_execute = bind_at_execute || found->second.bind_at_execute;
        is_persistent = is_persistent || found->second.is_persistent_input;
    }

    MappedTensor mapped{
        node,
        nullptr,
        node->dtype(),
        static_cast<std::size_t>(node->nelems()),
        needs_host,
        bind_at_execute,
        is_persistent,
        host_ptr};
    write_mapped_tensor_locked(impl_key, mapped);
    at::Tensor mutable_tensor = tensor;
    if (found != g_tensor_nodes.end())
    {
        found->second.node = node;
        found->second.dtype = node->dtype();
        found->second.count = static_cast<std::size_t>(node->nelems());
    }
    if (nntile_binding(mutable_tensor) == nullptr)
    {
        attach_node_binding(mutable_tensor, node, mapped);
    }
    assert_has_node_ref(tensor, "register_data_node");
}

nntile::TensorGraph::TensorNode *lookup_data_node(TensorImplKey impl_key)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found != g_tensor_nodes.end())
    {
        return found->second.node;
    }
    return nullptr;
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
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end() || found->second.node == nullptr)
    {
        return nullptr;
    }
    if (found->second.node->graph() != g_graph.get() ||
        graph_numel(found->second.node->shape()) != graph_numel(shape))
    {
        g_tensor_nodes.erase(found);
        return nullptr;
    }
    ensure_meta_from_mapped_locked(tensor, found->second);
    return ensure_graph_shape_bridge_locked(found->second.node, shape);
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
    unregister_binding_impl(key);
    g_tensor_nodes.erase(key);
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
    nntile::TensorGraph::TensorNode *src_node = nntile_node(self);
    if (src_node == nullptr && source != nullptr)
    {
        src_node = source->node;
    }
    if (src_node == nullptr)
    {
        return;
    }

    const nntile::DataType dtype = source != nullptr ? source->dtype : src_node->dtype();
    if (graph_numel(src_node->shape()) != graph_numel(view_shape))
    {
        throw std::invalid_argument(
            "view: storage alias must preserve numel");
    }

    MappedTensor updated = source != nullptr ? *source : MappedTensor{};
    updated.node = src_node;
    updated.dtype = dtype;
    updated.count = static_cast<std::size_t>(graph_numel(view_shape));
    if (view_key != self_key)
    {
        write_mapped_tensor_locked(view_key, updated);
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

bool is_tensor_graph_output(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    if (nntile::TensorGraph::TensorNode *node = nntile_node(tensor);
        node != nullptr)
    {
        return node->is_output();
    }
    const TensorImplKey impl_key = canonical_tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end() || found->second.node == nullptr)
    {
        return false;
    }
    return found->second.node->is_output();
}

void stage_tensor_for_axis_group_compile(const at::Tensor &tensor)
{
    if (is_tensor_graph_output(tensor))
    {
        return;
    }
    at::Tensor mutable_tensor = tensor;
    if (is_metadata_only_tensor(mutable_tensor))
    {
        pin_tensor_for_graph(mutable_tensor);
        return;
    }
    if (!has_host_staging(mutable_tensor))
    {
        ensure_host_staging(mutable_tensor);
    }
    if (has_host_staging(mutable_tensor))
    {
        mark_staged_input_tensor(mutable_tensor);
    }
}

void refresh_staged_tensor_mapping(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end())
    {
        return;
    }
    found->second.bind_at_execute = true;
    if (has_host_staging(tensor))
    {
        found->second.host_data_ptr = tensor.storage().data_ptr().get();
    }
}

void mark_persistent_graph_tensor(const at::Tensor &tensor)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    const TensorImplKey impl_key = tensor_impl_key(tensor);
    const auto found = g_tensor_nodes.find(impl_key);
    if (found == g_tensor_nodes.end())
    {
        MappedTensor mapped{
            nullptr,
            nullptr,
            nntile::DataType::FP32,
            0,
            false,
            false,
            true,
            nullptr};
        write_mapped_tensor_locked(impl_key, mapped);
    }
    else
    {
        found->second.is_persistent_input = true;
        found->second.needs_host_copy = true;
        found->second.bind_at_execute = true;
    }
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
    stats.tensor_nodes = static_cast<std::int64_t>(g_tensor_nodes.size());
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

bool is_tensor_graph_output(const at::Tensor & /*tensor*/)
{
    require_libnntile();
    return false;
}

void stage_tensor_for_axis_group_compile(const at::Tensor & /*tensor*/)
{
    require_libnntile();
}

void refresh_staged_tensor_mapping(const at::Tensor & /*tensor*/)
{
}

void mark_persistent_graph_tensor(const at::Tensor & /*tensor*/)
{
}

bool read_nntile_staging_to_host(const at::Tensor & /*tensor*/, void * /*host_ptr*/)
{
    return false;
}

bool read_nntile_logical_to_host(const at::Tensor & /*tensor*/, void * /*host_ptr*/)
{
    return false;
}

const std::int64_t *label_host_cache_ptr(
    const at::Tensor & /*tensor*/,
    std::size_t * /*out_count*/)
{
    return nullptr;
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
