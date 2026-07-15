/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_tensor_meta.h
 * Graph binding metadata on device=nntile TensorImpl.
 */

#pragma once

#include <c10/core/TensorImpl.h>

#ifdef TORCH_NNTILE_USE_LIBNNTILE
#include <nntile/tensor/tensor_ref.hh>
#endif

namespace at
{
class Tensor;
}

namespace torch_nntile
{

#ifdef TORCH_NNTILE_USE_LIBNNTILE

struct NNTileBackendMeta final : c10::BackendMeta
{
    nntile::TensorRef ref;

    explicit NNTileBackendMeta(nntile::TensorRef ref_in);

    c10::intrusive_ptr<c10::BackendMeta> clone(
        const c10::intrusive_ptr<c10::BackendMeta> &ptr) const override;
};

void assert_has_tensor_ref(const at::Tensor &tensor, const char *site);

nntile::TensorRef tensor_ref(const at::Tensor &tensor);

nntile::TensorGraph::TensorNode *nntile_node(const at::Tensor &tensor);

void attach_tensor_ref(at::Tensor &tensor, nntile::TensorRef ref);

void share_tensor_ref_for_reshape(const at::Tensor &base, at::Tensor &view);

//! True while the recorder TensorGraph (and its TensorNodes) are alive.
bool logical_tensor_nodes_alive();

//! Called by the graph recorder around TensorGraph create/destroy.
void set_logical_tensor_nodes_alive(bool alive);

//! Legacy release note (drained at compile/wait; reclaim is graph INVALIDATE
//! from ``TensorRef`` last-drop, not a side-channel flush).
void note_logical_released(nntile::TensorGraph::TensorNode *logical);

//! Drain ``note_logical_released`` / ``note_tensor_ref_released`` queue
//! (call under recorder lock). Does not ``invalidate_logical_tiles`` —
//! reclaim is ordinary graph ``INVALIDATE`` only.
std::vector<nntile::TensorGraph::TensorNode *> take_released_logicals();

#endif // TORCH_NNTILE_USE_LIBNNTILE

} // namespace torch_nntile
