/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_meta.cpp
 */

#include "nntile_tensor_meta.h"

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE

#include <atomic>
#include <cstdlib>

namespace torch_nntile
{

namespace
{

std::atomic<bool> g_logical_tensor_nodes_alive{false};

bool trace_assert_enabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv("TORCH_NNTILE_ASSERT_NODE_REF");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

NNTileBackendMeta *backend_meta_ptr(const at::Tensor &tensor)
{
    if (tensor.device().type() != c10::DeviceType::PrivateUse1)
    {
        return nullptr;
    }
    return static_cast<NNTileBackendMeta *>(
        tensor.unsafeGetTensorImpl()->get_backend_meta());
}

} // namespace

bool logical_tensor_nodes_alive()
{
    return g_logical_tensor_nodes_alive.load(std::memory_order_acquire);
}

void set_logical_tensor_nodes_alive(bool alive)
{
    g_logical_tensor_nodes_alive.store(alive, std::memory_order_release);
}

NNTileBinding::NNTileBinding(nntile::TensorGraph::TensorNode *logical_in)
    : logical(logical_in)
{
    if (logical != nullptr && logical_tensor_nodes_alive())
    {
        logical->mark_output(true);
    }
}

NNTileBinding::~NNTileBinding()
{
    // TensorGraph may already be destroyed (atexit / reset_graph_session).
    // Skip mark_output on a stale node pointer.
    if (logical != nullptr && logical_tensor_nodes_alive())
    {
        logical->mark_output(false);
    }
    logical = nullptr;
}

NNTileBackendMeta::NNTileBackendMeta(NodeRef binding_in)
    : binding(std::move(binding_in))
{
}

c10::intrusive_ptr<c10::BackendMeta> NNTileBackendMeta::clone(
    const c10::intrusive_ptr<c10::BackendMeta> &ptr) const
{
    const auto *other = static_cast<const NNTileBackendMeta *>(ptr.get());
    return c10::make_intrusive<NNTileBackendMeta>(other->binding);
}

void assert_has_node_ref(const at::Tensor &tensor, const char *site)
{
    if (!trace_assert_enabled())
    {
        return;
    }
    TORCH_CHECK(
        nntile_binding(tensor) != nullptr,
        "TORCH_NNTILE_ASSERT_NODE_REF: missing NodeRef at ",
        site);
}

NodeRef nntile_binding(const at::Tensor &tensor)
{
    NNTileBackendMeta *meta = backend_meta_ptr(tensor);
    if (meta == nullptr)
    {
        return nullptr;
    }
    return meta->binding;
}

nntile::TensorGraph::TensorNode *nntile_node(const at::Tensor &tensor)
{
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr)
    {
        return nullptr;
    }
    return binding->logical;
}

void attach_binding(at::Tensor &tensor, NodeRef binding)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "attach_binding: expected nntile tensor");
    auto meta = c10::make_intrusive<NNTileBackendMeta>(std::move(binding));
    c10::TensorImpl *impl = tensor.unsafeGetTensorImpl();
    impl->set_backend_meta(std::move(meta));
}

void share_node_ref_for_reshape(const at::Tensor &base, at::Tensor &view)
{
    NodeRef binding = nntile_binding(base);
    if (binding == nullptr)
    {
        return;
    }
    attach_binding(view, binding);
}

} // namespace torch_nntile

#endif // TORCH_NNTILE_USE_LIBNNTILE
