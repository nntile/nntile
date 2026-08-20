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
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <stdexcept>

#include <nntile/runtime.hh>
#include <nntile/starpu/sync_defer.hh>
#include <nntile/dtype.hh>
#include <nntile/tensor/axis_descriptor.hh>
#include <nntile/tensor/graph.hh>
#include <nntile/tensor/ops/clear.hh>
#include <nntile/tensor/ops/invalidate.hh>
#include <nntile/tensor/ops/unregister.hh>
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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch_nntile
{

namespace
{

//! Temporary profiling knob: skip StarPU *compute* submit and host<->tile
//! acquire/memcpy so record+compile wall time can be measured alone.
//! ``execute_range`` still runs (watermark + last-consumer reclaim) so
//! incremental compile stays O(pending). Set ``TORCH_NNTILE_SKIP_STARPU=1``.
//! Results / accuracy are meaningless.
bool skip_starpu_submit_and_acquire()
{
    static int const cached = []() -> int
    {
        char const *env = std::getenv("TORCH_NNTILE_SKIP_STARPU");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0)
        {
            return 0;
        }
        return 1;
    }();
    return cached != 0;
}

std::recursive_mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
struct ParamGradEntry
{
    nntile::TensorGraph::TensorNode *grad_node = nullptr;
    at::Tensor param;
};
std::unordered_map<TensorImplKey, ParamGradEntry> g_param_grad_registry;
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
    //! Post-DCE execution_order index already submitted via execute_range.
    std::size_t executed_op_end = 0;
    //! Slice scheduled by the latest compile_graph_locked call.
    std::size_t pending_exec_op_begin = 0;
    std::size_t pending_exec_op_end = 0;
};

std::unique_ptr<RecorderExecState> g_exec;
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
    // HF layout tax: views on CUDA, materializing copies on nntile.
    std::uint64_t record_narrow_copy_calls = 0;
    std::uint64_t record_transpose_copy_calls = 0;
    std::uint64_t record_narrow_copy_elems = 0;
    std::uint64_t record_transpose_copy_elems = 0;
};

GraphApiTimingStats g_timing;

using SteadyClock = std::chrono::steady_clock;

double seconds_since(SteadyClock::time_point const start)
{
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

void reclaim_pending_outputs_locked()
{
    // Replaced by TensorGraph INVALIDATE ops appended at compile.
}

void collect_pending_output_reclaim_locked(
    const nntile::TensorGraph::PhaseSnapshot & /*phase*/)
{
}

void sync_param_grad_aliases_locked();

void compile_graph_locked();

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

//! Temporary: PrivateUse1 aten ops require a single tile per tensor while
//! torch-native StarPU codelets are introduced. See
//! docs/dev/torch_starpu_kernels.md.
[[noreturn]] void throw_tiled_aten_temporarily_disabled()
{
    throw std::runtime_error(
        "torch_nntile: axis-group tiling is temporarily disabled for "
        "device=nntile PrivateUse1 aten ops (single-tile / untiled tensors "
        "only). See docs/dev/torch_nntile_aten_ops.md and "
        "docs/dev/torch_starpu_kernels.md.");
}

void require_untiled_torch_session_locked()
{
    if (g_graph == nullptr)
    {
        return;
    }
    for (nntile::AxisDescriptor *axis : g_graph->axis_groups())
    {
        if (axis != nullptr && axis->is_tiled())
        {
            throw_tiled_aten_temporarily_disabled();
        }
    }
}

void apply_pending_axis_tiling_locked()
{
    if (g_graph == nullptr || g_axis_tiling_by_name.empty())
    {
        return;
    }
    // Reject before mutating AxisDescriptors.
    throw_tiled_aten_temporarily_disabled();
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
    const auto *tiles = g_exec->tile_map.try_get(staging);
    if (tiles == nullptr || tiles->size() != 1)
    {
        throw std::runtime_error(
            "torch_nntile: io_staging must be single-tile");
    }
    nntile::TileGraph::TileNode *tile = (*tiles)[0];
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
    bool const skip_starpu = skip_starpu_submit_and_acquire();
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
        if (!skip_starpu)
        {
            auto local = buf.acquire(STARPU_W);
            const auto *src = static_cast<const float *>(host_ptr);
            for (std::size_t i = 0; i < count; ++i)
            {
                local[static_cast<nntile::Index>(i)] =
                    nntile::fp32_t(src[i]);
            }
            local.release();
        }
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
        if (!skip_starpu)
        {
            auto local = buf.acquire(STARPU_W);
            const auto *src =
                static_cast<const std::int64_t *>(host_ptr);
            for (std::size_t i = 0; i < count; ++i)
            {
                local[static_cast<nntile::Index>(i)] =
                    nntile::int64_t(src[i]);
            }
            local.release();
        }
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
        if (!skip_starpu)
        {
            auto local = buf.acquire(STARPU_W);
            // Torch bool / uint8 storage is byte-addressed.
            const auto *src =
                static_cast<const std::uint8_t *>(host_ptr);
            for (std::size_t i = 0; i < count; ++i)
            {
                local[static_cast<nntile::Index>(i)] =
                    nntile::bool_t(src[i] != 0);
            }
            local.release();
        }
        break;
    }
    default:
        throw std::runtime_error(
            "torch_nntile: unsupported staging write dtype");
    }
    runtime.mark_initialized(staging);
}

void release_io_staging_locked(nntile::TensorGraph::TensorNode *staging)
{
    if (staging == nullptr || g_exec == nullptr || g_exec->runtime == nullptr)
    {
        return;
    }
    // Clear logical marks, then drop StarPU tile payloads
    // (invalidate_submit alone left handles live).
    g_exec->runtime->invalidate_logical_tiles(staging);
    g_exec->inc_state.tensor_to_tiles.erase(staging);
    g_exec->inc_state.tensor_layout_fp.erase(staging);
    g_exec->tile_map.erase(staging);
    if (g_exec->session_tiling != nullptr)
    {
        g_exec->session_tiling->erase(staging);
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
    if (skip_starpu_submit_and_acquire())
    {
        return;
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
            dst[i] = static_cast<float>(
                local[static_cast<nntile::Index>(i)]);
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
        // Torch bool storage is 1 byte/elem; do not write through bool*
        // (pointer arithmetic / aliasing). Treat host buffer as bytes.
        auto *dst = static_cast<std::uint8_t *>(host_ptr);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<std::uint8_t>(
                static_cast<bool>(local[static_cast<nntile::Index>(i)]));
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
    auto *staging = g_graph->emplace_data(logical->shape(), logical->dtype());
    staging->set_name(
        std::string("io_staging_") + logical->name() + "_" + tag + "_" +
        std::to_string(++g_ephemeral_staging_serial));
    return staging;
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

//! True when ``view`` indices stay inside a storage of ``storage_numel``
//! elements. Allows broadcast ``expand`` (zero strides) whose logical
//! numel exceeds storage numel — unlike a raw numel comparison.
bool view_fits_storage(
    const at::Tensor &view,
    nntile::Index storage_numel)
{
    if (storage_numel < 0)
    {
        return false;
    }
    if (view.numel() == 0)
    {
        return view.storage_offset() >= 0
            && view.storage_offset() <= storage_numel;
    }
    const int64_t offset = view.storage_offset();
    if (offset < 0)
    {
        return false;
    }
    int64_t max_index = offset;
    int64_t min_index = offset;
    const auto sizes = view.sizes();
    const auto strides = view.strides();
    for (int64_t i = 0; i < sizes.size(); ++i)
    {
        const int64_t size = sizes[i];
        const int64_t stride = strides[i];
        if (size <= 0)
        {
            continue;
        }
        const int64_t span = (size - 1) * stride;
        if (span >= 0)
        {
            max_index += span;
        }
        else
        {
            min_index += span;
        }
    }
    return min_index >= 0 && max_index < storage_numel;
}

//! Bytes available in ``tensor`` storage from ``data_ptr()`` to the end.
//!
//! ``tensor.nbytes()`` is typically ``numel * itemsize`` and does **not**
//! catch a contiguous view whose ``storage_offset`` leaves too little room
//! (``batch[:, 1:]`` at B=1 is contiguous with offset != 0).
std::size_t host_bytes_from_data_ptr(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.is_cpu(), "host_bytes_from_data_ptr: CPU tensor");
    const int64_t itemsize = tensor.element_size();
    TORCH_CHECK(itemsize > 0, "host_bytes_from_data_ptr: bad itemsize");
    const int64_t offset = tensor.storage_offset();
    TORCH_CHECK(offset >= 0, "host_bytes_from_data_ptr: negative offset");
    const int64_t storage_bytes =
        static_cast<int64_t>(tensor.storage().nbytes());
    const int64_t offset_bytes = offset * itemsize;
    TORCH_CHECK(
        offset_bytes <= storage_bytes,
        "host_bytes_from_data_ptr: storage_offset past end of storage "
        "(offset=",
        offset,
        " storage_bytes=",
        storage_bytes,
        " itemsize=",
        itemsize,
        ")");
    return static_cast<std::size_t>(storage_bytes - offset_bytes);
}

void check_host_buffer_for_transfer(
    const at::Tensor &host,
    std::size_t count,
    std::size_t elem_bytes,
    const char *what)
{
    TORCH_CHECK(host.is_cpu(), what, ": expected CPU tensor");
    TORCH_CHECK(host.is_contiguous(), what, ": contiguous required");
    TORCH_CHECK(
        static_cast<std::size_t>(host.numel()) == count,
        what,
        ": numel mismatch (numel=",
        host.numel(),
        " count=",
        count,
        " storage_offset=",
        host.storage_offset(),
        ")");
    const std::size_t need = count * elem_bytes;
    const std::size_t avail = host_bytes_from_data_ptr(host);
    TORCH_CHECK(
        avail >= need,
        what,
        ": storage too small from data_ptr() (avail=",
        avail,
        " need=",
        need,
        " storage_offset=",
        host.storage_offset(),
        " storage.nbytes=",
        host.storage().nbytes(),
        ")");
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
    auto *view_node = g_graph->emplace_data(shape, node->dtype());
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
    if (nntile::TensorRef binding = tensor_ref(mutable_tensor);
        binding)
    {
        nntile::TensorGraph::TensorNode *logical = binding.get();
        if (logical->graph() != g_graph.get())
        {
            throw std::runtime_error(
                "torch_nntile: tensor logical node does not belong to the "
                "active TensorGraph");
        }
        const nntile::Index storage_n = graph_numel(logical->shape());
        // Views (transpose/narrow/split/expand) share the parent storage
        // node. Layout (sizes/strides/offset) is packed into
        // TorchDispatchArgs at record time — do not densify here.
        // Broadcast expand may have larger logical numel than storage.
        if (!view_fits_storage(mutable_tensor, storage_n))
        {
            throw std::invalid_argument(
                "torch_nntile: view indices exceed storage logical");
        }
        return logical;
    }

    // Unbound non-dense views must share a parent via as_strided/alias/
    // narrow. Inventing a fresh node of torch.numel() would rebind a
    // packed QKV slice to a too-small logical (RoPE rotate_half).
    if (!mutable_tensor.is_contiguous() ||
        mutable_tensor.storage_offset() != 0)
    {
        throw std::runtime_error(
            "torch_nntile: unbound non-dense view (missing alias/"
            "as_strided TensorRef share)");
    }

    nntile::TensorRef node_ref = g_graph->data(shape, dtype);
    nntile::TensorGraph::TensorNode *node = node_ref.get();
    apply_axis_name_hints_locked(impl_key, node);

    if (!static_cast<bool>(tensor_ref(mutable_tensor)))
    {
        attach_tensor_ref(mutable_tensor, std::move(node_ref));
    }

    return node;
}

void clear_pending_recorder_state_locked()
{
    clear_param_grad_registry_locked();
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

void compact_tensor_graph_session_locked()
{
    // Drop sealed TensorGraph ops so the next record/compile is O(phase).
    // Unsealed ops recorded after the last seal (next phase already in
    // flight while a prior run() completes) are preserved.
    // Tech debt D1: TensorNode / TileNode IR is not cleared; only ops.
    if (g_graph == nullptr)
    {
        return;
    }
    g_graph->drop_all_ops();
    // Mirror TensorGraph compact on the tile side: when every compiled tile
    // op has finished, drop TileGraph ops + Runtime execution_order_ so
    // session history does not grow with step count. Tile nodes / payloads
    // stay (weights, live activations). Must clear TileGraph::ops whenever
    // Runtime resets compiled_graph_op_count_, or the next compile() would
    // re-append the entire historical list.
    if (g_exec != nullptr &&
        g_exec->runtime != nullptr &&
        g_exec->tile_graph != nullptr &&
        g_exec->runtime->drop_fully_executed_history())
    {
        g_exec->tile_graph->clear_ops();
        g_exec->executed_op_end = 0;
        g_exec->pending_exec_op_begin = 0;
        g_exec->pending_exec_op_end = 0;
    }
}

//! Clear recorder side state so only Python-held TensorRefs remain live
//! for INVALIDATE selection. Does not wait on StarPU.
void prepare_invalidate_selection_locked()
{
    clear_pending_recorder_state_locked();
}

//! Drain the TensorRef-release queue without touching StarPU.
//!
//! Reclaim is **only** via ordinary graph ops: ``TensorRef`` last-drop
//! records ``tensor::unregister``, and
//! ``append_invalidates_for_unmarked_unsealed`` covers emplace_data temps.
//! A side-channel ``invalidate_logical_tiles`` here ran *before* the phase
//! was submitted, so ``del inputs`` after record (but before compile) could
//! free ingress tiles before embedding tasks were inserted - StarPU then
//! saw "handle is not initialized".
void flush_released_logicals_locked()
{
    (void) take_released_logicals();
}

void compact_after_submit_locked()
{
    compact_tensor_graph_session_locked();
}

void compile_graph_locked()
{
    // Do not wait for a prior async run(): sealing / lowering the next
    // phase while StarPU still executes the previous one is allowed.
    // Unmarked phase temps become TensorGraph INVALIDATE ops (async submit).

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
    require_untiled_torch_session_locked();

    // Marks must reflect live Python refs before INVALIDATE selection.
    prepare_invalidate_selection_locked();
    // Drain release notes only; do not invalidate_logical_tiles here.
    // Payload reclaim is append_invalidates + TensorRef-recorded
    // UNREGISTER ops in this phase (submitted with compute so StarPU
    // orders them).
    flush_released_logicals_locked();
    nntile::tensor::append_invalidates_for_unmarked_unsealed(*g_graph);

    SteadyClock::time_point t_part = SteadyClock::now();
    const nntile::TensorGraph::PhaseSnapshot phase = g_graph->seal_phase();
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
    // Submit the pending tile-op range. INVALIDATE / UNREGISTER ops are
    // already in the execution stream. Join StarPU via wait() for readout.
    if (g_exec->pending_exec_op_end > g_exec->pending_exec_op_begin)
    {
        std::uint64_t const phase_ops = static_cast<std::uint64_t>(
            g_exec->pending_exec_op_end - g_exec->pending_exec_op_begin);
        SteadyClock::time_point const t0 = SteadyClock::now();
        bool const submit = !skip_starpu_submit_and_acquire();
        // Always call execute_range so Runtime::executed_op_end_ advances.
        // SKIP_STARPU only disables OpNode::execute (StarPU task insert).
        g_exec->runtime->execute_range(
            g_exec->pending_exec_op_begin,
            g_exec->pending_exec_op_end,
            submit);
        g_timing.run_s += seconds_since(t0);
        ++g_timing.run_calls;
        g_timing.run_ops += phase_ops;
        g_exec->executed_op_end = g_exec->pending_exec_op_end;
        g_exec->pending_exec_op_begin = g_exec->pending_exec_op_end;
    }
    g_run_cleanup_pending = true;
    compact_after_submit_locked();
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
    // Join StarPU for host-visible completion. Session compact already ran
    // at the end of run(); reclaim ops were submitted with the phase.
    g_exec->runtime->wait();
    compact_tensor_graph_session_locked();
    // Drain release notes; reclaim ops were already submitted with the
    // phase (or recorded into the next unsealed phase on TensorRef drop).
    flush_released_logicals_locked();
    g_run_cleanup_pending = false;
    g_timing.wait_s += seconds_since(t0);
    ++g_timing.wait_calls;
}

void reset_recorder_locked(bool clear_tensor_gc)
{
    if (g_exec != nullptr && g_exec->runtime != nullptr)
    {
        g_exec->runtime->wait();
    }
    g_run_cleanup_pending = false;
    clear_pending_recorder_state_locked();
    g_ephemeral_staging_serial = 0;
    g_exec.reset();
    set_logical_tensor_nodes_alive(false);
    g_graph.reset();
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
    if (grad_node == nullptr)
    {
        return;
    }
    if (!static_cast<bool>(tensor_ref(grad)))
    {
        attach_tensor_ref(grad, nntile::TensorRef::adopt(grad_node));
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

void execute_pending_graph_locked()
{
    // compile + run only. Never wait here - callers must use wait() /
    // wait_graph_session() (same contract as compile_graph + run).
    compile_graph_locked();
    run_graph_locked();
}

void shutdown_recorder_locked()
{
    reset_recorder_locked(true);
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
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    execute_pending_graph_locked();
}

void compile_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    compile_graph_locked();
}

void run_graph()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    run_graph_locked();
}

void wait_graph_session()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    // finish_run_locked() already joins StarPU when a run is pending.
    finish_run_locked();
}

void reset_graph_session()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    reset_recorder_locked(true);
}

void shutdown_recorder()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    shutdown_recorder_locked();
}

bool has_graph_session()
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    return g_exec != nullptr && g_exec->runtime != nullptr;
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
    // Host readout must join a prior async run() before recording gather:
    // wait() compacts sealed history; without it, drop_all_ops on a later
    // wait could race with an in-flight phase that still owns these nodes.
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
    // Keep a TensorRef so compile's unmarked-INVALIDATE pass does not free
    // staging in the same phase as gather (before the host read below).
    nntile::TensorRef staging_hold = nntile::TensorRef::adopt(staging);
    // Output S: single-tile only; lowered during compile after gather is recorded.
    nntile::tensor::clear(staging);
    nntile::tensor::gather(logical, staging);

    compile_graph_locked();
    run_graph_locked();
    finish_run_locked();

    read_staging_to_host_locked(staging, host_ptr, dtype, count);
    // Drop hold without recording graph INVALIDATE (manual reclaim below).
    nntile::set_tensor_nodes_alive(false);
    staging_hold = nntile::TensorRef{};
    nntile::set_tensor_nodes_alive(true);
    release_io_staging_locked(staging);
    g_timing.host_readout_s += seconds_since(t0);
    ++g_timing.host_readout_calls;
}

void copy_nntile_tensor_to_cpu(const at::Tensor &src, at::Tensor &dst)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    nntile::TensorRef binding = tensor_ref(src);
    if (!binding)
    {
        throw std::runtime_error(
            "torch_nntile: copy nntile tensor to CPU requires a bound "
            "logical graph node (use .to('nntile') first)");
    }
    nntile::TensorGraph::TensorNode *logical = binding.get();
    const nntile::DataType dtype = logical->dtype();
    const std::size_t count =
        static_cast<std::size_t>(logical->nelems());
    const std::size_t elem_bytes = nntile::dtype_size(dtype);
    // Host buffer must match the logical payload size, including room
    // after storage_offset (dst.nbytes() alone is not enough).
    check_host_buffer_for_transfer(
        dst,
        count,
        elem_bytes,
        "torch_nntile: copy_nntile_tensor_to_cpu");
    TORCH_CHECK(
        aten_scalar_to_nntile_dtype(dst.scalar_type()) == dtype,
        "torch_nntile: host readout dtype mismatch");
    // Respect dst storage_offset (matches CPU->nntile ingress via data_ptr()).
    void *host_ptr = dst.data_ptr();

    // Sync a prior async execute()/run() even when no ops are pending so
    // subsequent gather recording is not wiped by wait-side drop_all_ops().
    if (g_run_cleanup_pending)
    {
        finish_run_locked();
    }

    if (g_graph != nullptr &&
        g_graph->num_ops() > g_graph->phase_seal_cursor())
    {
        compile_graph_locked();
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

at::Tensor gather_nntile_view_to_cpu(const at::Tensor &src)
{
    TORCH_CHECK(
        src.device().type() == c10::DeviceType::PrivateUse1,
        "gather_nntile_view_to_cpu: expected nntile");
    nntile::TensorRef binding = tensor_ref(src);
    TORCH_CHECK(
        binding,
        "gather_nntile_view_to_cpu: unbound tensor");
    nntile::TensorGraph::TensorNode *logical = binding.get();
    const bool dense_cover =
        src.is_contiguous() &&
        src.storage_offset() == 0 &&
        static_cast<int64_t>(logical->nelems()) == src.numel();
    if (dense_cover)
    {
        at::Tensor cpu = at::empty(
            src.sizes(),
            src.options().device(at::kCPU).memory_format(
                at::MemoryFormat::Contiguous));
        copy_nntile_tensor_to_cpu(src, cpu);
        return cpu;
    }
    std::vector<int64_t> full_sizes(
        logical->shape().begin(),
        logical->shape().end());
    if (full_sizes.empty())
    {
        full_sizes.push_back(static_cast<int64_t>(logical->nelems()));
    }
    at::Tensor full_cpu = at::empty(
        full_sizes,
        src.options().device(at::kCPU).memory_format(
            at::MemoryFormat::Contiguous));
    copy_nntile_tensor_to_cpu(src, full_cpu);
    return full_cpu.as_strided(
                   src.sizes(),
                   src.strides(),
                   src.storage_offset())
        .contiguous();
}

at::Tensor gather_full_logical_to_cpu(const at::Tensor &src)
{
    TORCH_CHECK(
        src.device().type() == c10::DeviceType::PrivateUse1,
        "gather_full_logical_to_cpu: expected nntile");
    nntile::TensorRef binding = tensor_ref(src);
    TORCH_CHECK(
        binding,
        "gather_full_logical_to_cpu: unbound tensor");
    nntile::TensorGraph::TensorNode *logical = binding.get();
    std::vector<int64_t> full_sizes(
        logical->shape().begin(),
        logical->shape().end());
    if (full_sizes.empty())
    {
        full_sizes.push_back(static_cast<int64_t>(logical->nelems()));
    }
    at::Tensor full_cpu = at::empty(
        full_sizes,
        src.options().device(at::kCPU).memory_format(
            at::MemoryFormat::Contiguous));
    copy_nntile_tensor_to_cpu(src, full_cpu);
    return full_cpu;
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
    TORCH_CHECK(
        cpu_src.scalar_type() == nntile_dst.scalar_type(),
        "init_nntile_input_from_cpu: dtype mismatch");
    const nntile::DataType dtype =
        aten_scalar_to_nntile_dtype(cpu_src.scalar_type());
    check_host_buffer_for_transfer(
        cpu_src,
        static_cast<std::size_t>(cpu_src.numel()),
        nntile::dtype_size(dtype),
        "init_nntile_input_from_cpu");

    if (g_graph == nullptr)
    {
        g_graph = std::make_unique<nntile::TensorGraph>("torch_nntile");
        set_logical_tensor_nodes_alive(true);
    }

    const std::vector<nntile::Index> shape =
        aten_sizes_to_graph_shape(cpu_src.sizes());
    const TensorImplKey impl_key = tensor_impl_key(nntile_dst);

    if (nntile::TensorRef existing = tensor_ref(nntile_dst);
        existing)
    {
        throw std::runtime_error(
            "torch_nntile: CPU->nntile copy into an already-bound tensor is "
            "unsupported; ingress each tensor once via .to('nntile')");
    }

    nntile::TensorRef logical_ref = g_graph->data(shape, dtype);
    nntile::TensorGraph::TensorNode *logical = logical_ref.get();
    apply_axis_name_hints_locked(impl_key, logical);

    attach_tensor_ref(nntile_dst, std::move(logical_ref));

    auto *staging = new_ephemeral_staging_node_locked(logical, "ingress");
    if (staging == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: failed to create ingress staging tensor");
    }
    lower_io_staging_locked(staging);
    // Use data_ptr() (not storage().data_ptr()): size-1 dims can be
    // is_contiguous() with storage_offset != 0 (e.g. batch[:, 1:] at B=1).
    write_cpu_bytes_to_staging_locked(
        staging,
        cpu_src.data_ptr(),
        dtype,
        static_cast<std::size_t>(cpu_src.numel()));

    nntile::tensor::scatter(staging, logical);
    // StarPU orders unregister after scatter. Peak ~2x during batched
    // .to("nntile") is accepted (same as model.cuda() while CPU lives).
    nntile::tensor::unregister(staging);
}

void overwrite_bound_nntile_logical_from_cpu(
    const at::Tensor &cpu_src,
    const at::Tensor &nntile_bound)
{
    std::lock_guard<std::recursive_mutex> lock(g_recorder_mutex);
    TORCH_CHECK(
        cpu_src.is_cpu(),
        "overwrite_bound_nntile_logical_from_cpu: expected CPU src");
    TORCH_CHECK(
        cpu_src.is_contiguous(),
        "overwrite_bound_nntile_logical_from_cpu: CPU src must be "
        "contiguous");
    TORCH_CHECK(
        nntile_bound.device().type() == c10::DeviceType::PrivateUse1,
        "overwrite_bound_nntile_logical_from_cpu: expected nntile");
    nntile::TensorRef binding = tensor_ref(nntile_bound);
    TORCH_CHECK(
        binding,
        "overwrite_bound_nntile_logical_from_cpu: unbound tensor");
    nntile::TensorGraph::TensorNode *logical = binding.get();
    const nntile::DataType dtype = logical->dtype();
    TORCH_CHECK(
        aten_scalar_to_nntile_dtype(cpu_src.scalar_type()) == dtype,
        "overwrite_bound_nntile_logical_from_cpu: dtype mismatch");
    TORCH_CHECK(
        static_cast<std::size_t>(cpu_src.numel()) ==
            static_cast<std::size_t>(logical->nelems()),
        "overwrite_bound_nntile_logical_from_cpu: numel mismatch");
    check_host_buffer_for_transfer(
        cpu_src,
        static_cast<std::size_t>(cpu_src.numel()),
        nntile::dtype_size(dtype),
        "overwrite_bound_nntile_logical_from_cpu");

    // Sync prior async work so scatter appends to a clean phase (same
    // pattern as copy_nntile_tensor_to_cpu).
    if (g_run_cleanup_pending)
    {
        finish_run_locked();
    }
    if (g_graph != nullptr &&
        g_graph->num_ops() > g_graph->phase_seal_cursor())
    {
        compile_graph_locked();
        run_graph_locked();
        finish_run_locked();
    }

    auto *staging =
        new_ephemeral_staging_node_locked(logical, "overwrite");
    if (staging == nullptr)
    {
        throw std::runtime_error(
            "torch_nntile: failed to create overwrite staging tensor");
    }
    lower_io_staging_locked(staging);
    write_cpu_bytes_to_staging_locked(
        staging,
        cpu_src.data_ptr(),
        dtype,
        static_cast<std::size_t>(cpu_src.numel()));
    nntile::tensor::scatter(staging, logical);
    nntile::tensor::unregister(staging);
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
    assert_has_tensor_ref(tensor, "get_or_create_data_node");
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
    TORCH_CHECK(node != nullptr, "register_data_node: null node");
    TORCH_CHECK(
        static_cast<std::size_t>(tensor.numel()) ==
            static_cast<std::size_t>(node->nelems()),
        "torch_nntile: register_data_node size mismatch: torch numel=",
        tensor.numel(),
        " logical nelems=",
        node->nelems(),
        " torch shape=",
        tensor.sizes());
    // SSA inplace (add_/mul_/…) allocates a new TensorNode; rebind the
    // TensorRef so the next forward reads the updated logical. Skipping
    // when a ref already exists left parameters stuck on the old leaf
    // (stock SGD appeared to step but losses never changed).
    at::Tensor mutable_tensor = tensor;
    nntile::TensorRef current = tensor_ref(mutable_tensor);
    if (!current || current.get() != node)
    {
        attach_tensor_ref(
            mutable_tensor,
            nntile::TensorRef::adopt(node));
    }
    assert_has_tensor_ref(tensor, "register_data_node");
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

void note_record_narrow_copy(std::uint64_t nelems)
{
    ++g_timing.record_narrow_copy_calls;
    g_timing.record_narrow_copy_elems += nelems;
}

void note_record_transpose_copy(std::uint64_t nelems)
{
    ++g_timing.record_transpose_copy_calls;
    g_timing.record_transpose_copy_elems += nelems;
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
    // Do not pin: ``param.grad`` is Python-reachable; pinning would fight
    // ``zero_grad(set_to_none=True)`` until compile.
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
    const nntile::Index storage_n = graph_numel(src_node->shape());
    // Reject OOB aliases; allow expand/broadcast (zero strides) whose
    // logical numel exceeds storage numel.
    if (!view_fits_storage(view, storage_n))
    {
        throw std::invalid_argument(
            "view: alias indices exceed storage logical");
    }
    at::Tensor mutable_view = view;
    share_tensor_ref_for_reshape(self, mutable_view);
    assert_has_tensor_ref(view, "record_view_alias");
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
    return static_cast<bool>(tensor_ref(tensor));
}

void stage_tensor_for_axis_group_compile(const at::Tensor &tensor)
{
    (void)tensor;
}

void set_axis_group_tiling(
    const std::string &name,
    const std::vector<std::int64_t> &tile_sizes)
{
    (void)name;
    (void)tile_sizes;
    throw_tiled_aten_temporarily_disabled();
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
    if (skip_starpu_submit_and_acquire())
    {
        ss << "  NOTE: TORCH_NNTILE_SKIP_STARPU=1 "
              "(no compute submit / staging acquire; reclaim on)\n";
    }
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
    ss << "  starpu_task_wait_for_all: "
       << nntile::g_starpu_wait_for_all_count.load()
       << " calls (all sources; should stay flat between run()s if async)\n";
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
        if (g_timing.record_narrow_copy_calls > 0 ||
            g_timing.record_transpose_copy_calls > 0)
        {
            // Residual materializing layout copies (should be rare once
            // transpose/narrow/split are zero-copy views).
            const double narrow_gib = static_cast<double>(
                    g_timing.record_narrow_copy_elems) *
                4.0 / (1024.0 * 1024.0 * 1024.0);
            const double transpose_gib = static_cast<double>(
                    g_timing.record_transpose_copy_elems) *
                4.0 / (1024.0 * 1024.0 * 1024.0);
            ss << "    layout copies (should be ~0 with view metadata):\n";
            ss << "      NarrowCopy: " << g_timing.record_narrow_copy_calls
               << " ops, " << narrow_gib << " GiB fp32\n";
            ss << "      TransposeCopy: "
               << g_timing.record_transpose_copy_calls << " ops, "
               << transpose_gib << " GiB fp32\n";
        }
    }
    if (g_timing.run_calls > 0)
    {
        ss << "  note: wait_calls should be ~ run_calls when callers avoid "
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

