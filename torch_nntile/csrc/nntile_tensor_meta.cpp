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

#include <mutex>
#include <unordered_set>

namespace torch_nntile
{

namespace
{

std::mutex g_binding_mutex;
std::unordered_set<c10::TensorImpl *> g_live_binding_impls;

bool trace_assert_enabled()
{
    static const bool enabled = []() {
        const char *env = std::getenv("TORCH_NNTILE_ASSERT_NODE_REF");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

void register_binding_impl(c10::TensorImpl *impl)
{
    if (impl == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_binding_mutex);
    g_live_binding_impls.insert(impl);
}

NNTileBackendMeta *backend_meta_ptr(const at::Tensor &tensor)
{
    if (tensor.device().type() != c10::DeviceType::PrivateUse1)
    {
        return nullptr;
    }
    return static_cast<NNTileBackendMeta *>(tensor.unsafeGetTensorImpl()->get_backend_meta());
}

} // namespace

NNTileBinding::NNTileBinding(nntile::TensorGraph::TensorNode *logical_in)
    : logical(logical_in)
{
    if (logical != nullptr)
    {
        logical->mark_output(true);
    }
}

NNTileBinding::~NNTileBinding()
{
    if (logical != nullptr)
    {
        logical->mark_output(false);
    }
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

bool assert_node_ref_enabled()
{
    return trace_assert_enabled();
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

nntile::TensorGraph::TensorNode *nntile_io_staging(const at::Tensor &tensor)
{
    NodeRef binding = nntile_binding(tensor);
    if (binding == nullptr)
    {
        return nullptr;
    }
    return binding->io_staging;
}

void attach_binding(at::Tensor &tensor, NodeRef binding)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "attach_binding: expected nntile tensor");
    auto meta = c10::make_intrusive<NNTileBackendMeta>(std::move(binding));
    c10::TensorImpl *impl = tensor.unsafeGetTensorImpl();
    impl->set_backend_meta(std::move(meta));
    register_binding_impl(impl);
}

void unregister_binding_impl(c10::TensorImpl *impl)
{
    if (impl == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_binding_mutex);
    g_live_binding_impls.erase(impl);
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

std::int64_t count_live_bindings()
{
    std::lock_guard<std::mutex> lock(g_binding_mutex);
    return static_cast<std::int64_t>(g_live_binding_impls.size());
}

void clear_binding_registry()
{
    std::lock_guard<std::mutex> lock(g_binding_mutex);
    g_live_binding_impls.clear();
}

} // namespace torch_nntile

#endif // TORCH_NNTILE_USE_LIBNNTILE
