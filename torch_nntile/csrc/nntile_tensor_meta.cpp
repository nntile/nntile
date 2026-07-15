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

#include <cstdlib>

namespace torch_nntile
{

namespace
{

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
    return nntile::tensor_nodes_alive();
}

void set_logical_tensor_nodes_alive(bool alive)
{
    nntile::set_tensor_nodes_alive(alive);
}

void note_logical_released(nntile::TensorGraph::TensorNode *logical)
{
    nntile::note_tensor_ref_released(logical);
}

std::vector<nntile::TensorGraph::TensorNode *> take_released_logicals()
{
    return nntile::take_released_tensor_refs();
}

NNTileBackendMeta::NNTileBackendMeta(nntile::TensorRef ref_in)
    : ref(std::move(ref_in))
{
}

c10::intrusive_ptr<c10::BackendMeta> NNTileBackendMeta::clone(
    const c10::intrusive_ptr<c10::BackendMeta> &ptr) const
{
    const auto *other = static_cast<const NNTileBackendMeta *>(ptr.get());
    return c10::make_intrusive<NNTileBackendMeta>(other->ref);
}

void assert_has_tensor_ref(const at::Tensor &tensor, const char *site)
{
    if (!trace_assert_enabled())
    {
        return;
    }
    TORCH_CHECK(
        static_cast<bool>(tensor_ref(tensor)),
        "TORCH_NNTILE_ASSERT_NODE_REF: missing TensorRef at ",
        site);
}

nntile::TensorRef tensor_ref(const at::Tensor &tensor)
{
    NNTileBackendMeta *meta = backend_meta_ptr(tensor);
    if (meta == nullptr)
    {
        return nntile::TensorRef{};
    }
    return meta->ref;
}

nntile::TensorGraph::TensorNode *nntile_node(const at::Tensor &tensor)
{
    return tensor_ref(tensor).get();
}

void attach_tensor_ref(at::Tensor &tensor, nntile::TensorRef ref)
{
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "attach_tensor_ref: expected nntile tensor");
    auto meta = c10::make_intrusive<NNTileBackendMeta>(std::move(ref));
    c10::TensorImpl *impl = tensor.unsafeGetTensorImpl();
    impl->set_backend_meta(std::move(meta));
}

void share_tensor_ref_for_reshape(const at::Tensor &base, at::Tensor &view)
{
    nntile::TensorRef ref = tensor_ref(base);
    if (!ref)
    {
        return;
    }
    attach_tensor_ref(view, std::move(ref));
}

} // namespace torch_nntile

#endif // TORCH_NNTILE_USE_LIBNNTILE
