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
#include <nntile/tensor/graph.hh>
#include <nntile/tile/graph.hh>

#include <cstring>
#include <memory>
#include <mutex>
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
    nntile::DataType dtype = nntile::DataType::FP32;
    std::size_t count = 0;
    //! Copy runtime result into the PyTorch storage after execute.
    bool needs_host_copy = false;
    //! Bind PyTorch storage before execute (inputs and final outputs only).
    bool bind_at_execute = false;
    //! User/parameter inputs that must stay bound across multiple ops.
    bool is_persistent_input = false;
};

std::mutex g_recorder_mutex;
std::unique_ptr<nntile::TensorGraph> g_graph;
std::unordered_map<void *, MappedTensor> g_tensor_nodes;
std::unordered_set<nntile::TensorGraph::TensorNode *> g_all_nodes;
std::vector<at::Tensor> g_pinned_tensors;

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

void copy_output_if_needed(
    nntile::Runtime &runtime,
    const MappedTensor &mapped,
    void *data_ptr)
{
    const std::size_t count = mapped.count;
    if (count == 0 || !mapped.node->is_output())
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

void reset_recorder_locked()
{
    g_graph.reset();
    g_tensor_nodes.clear();
    g_all_nodes.clear();
    g_pinned_tensors.clear();
}

void execute_pending_graph_locked()
{
    if (g_graph == nullptr || g_graph->num_ops() == 0)
    {
        return;
    }

    ensure_nntile_context();

    // Mark every recorded node as output so DCE cannot drop it.
    for (nntile::TensorGraph::TensorNode *node : g_all_nodes)
    {
        node->mark_output(true);
    }

    nntile::TileGraph tile_graph =
        nntile::TileGraph::from_tensor_graph(*g_graph);
    nntile::Runtime runtime(tile_graph);
    runtime.compile();

    // Bind only storages that are still live (inputs and leaf outputs).
    // Intermediate buffers may already be freed by PyTorch.
    for (const auto &[data_ptr, mapped] : g_tensor_nodes)
    {
        if (!mapped.bind_at_execute)
        {
            continue;
        }
        mapped.node->mark_input(true);
        const std::size_t count = mapped.count;
        switch (mapped.dtype)
        {
        case nntile::DataType::FP32:
            runtime.bind_data(
                mapped.node,
                static_cast<const float *>(data_ptr),
                count);
            break;
        case nntile::DataType::INT64:
            runtime.bind_data(
                mapped.node,
                static_cast<const std::int64_t *>(data_ptr),
                count);
            break;
        default:
            throw std::runtime_error(
                "torch_nntile execute: unsupported bind dtype");
        }
    }

    runtime.execute();
    runtime.wait();

    copy_host_visible_outputs(runtime, nullptr);

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
    if (found != g_tensor_nodes.end())
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

    auto *node = g_graph->data(shape, dtype);
    if (mark_as_input)
    {
        node->mark_input(true);
    }
    track_node(node);
    g_tensor_nodes[data_ptr] = MappedTensor{
        node,
        dtype,
        static_cast<std::size_t>(graph_numel(shape)),
        false,
        mark_as_input,
        mark_as_input};
    return node;
}

void register_data_node(
    void *data_ptr,
    nntile::TensorGraph::TensorNode *node)
{
    std::lock_guard<std::mutex> lock(g_recorder_mutex);
    node->mark_output(true);
    track_node(node);
    g_tensor_nodes[data_ptr] = MappedTensor{
        node,
        node->dtype(),
        static_cast<std::size_t>(node->nelems()),
        true,
        true,
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

} // namespace torch_nntile

#endif
