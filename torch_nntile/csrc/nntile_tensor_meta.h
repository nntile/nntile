/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_meta.h
 * Graph binding metadata on device=nntile TensorImpl.
 */

#pragma once

#include <c10/core/TensorImpl.h>

#include <memory>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/tensor/graph.hh>
#endif

namespace at
{
class Tensor;
}

namespace torch_nntile
{

#ifdef TORCH_NNTILE_USE_LIBNNTILE

//! Graph binding for one at::Tensor: logical node L only (staging S is ephemeral).
struct NNTileBinding
{
    nntile::TensorGraph::TensorNode *logical = nullptr;

    explicit NNTileBinding(nntile::TensorGraph::TensorNode *logical_in);
    ~NNTileBinding();

    NNTileBinding(const NNTileBinding &) = delete;
    NNTileBinding &operator=(const NNTileBinding &) = delete;
};

using NodeRef = std::shared_ptr<NNTileBinding>;

struct NNTileBackendMeta final : c10::BackendMeta
{
    NodeRef binding;

    explicit NNTileBackendMeta(NodeRef binding_in);

    c10::intrusive_ptr<c10::BackendMeta> clone(
        const c10::intrusive_ptr<c10::BackendMeta> &ptr) const override;
};

void assert_has_node_ref(const at::Tensor &tensor, const char *site);

NodeRef nntile_binding(const at::Tensor &tensor);

nntile::TensorGraph::TensorNode *nntile_node(const at::Tensor &tensor);

void attach_binding(at::Tensor &tensor, NodeRef binding);

void share_node_ref_for_reshape(const at::Tensor &base, at::Tensor &view);

#endif // TORCH_NNTILE_USE_LIBNNTILE

} // namespace torch_nntile
