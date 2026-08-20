/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/tensor_ref.hh
 * External accessibility handle for a TensorGraph::TensorNode.
 * Last drop records async unregister; the IR node stays graph-owned.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <vector>

#include <nntile/tensor/graph_decl.hh>
#include <nntile/tensor/graph_data_node.hh>

namespace nntile
{

//! Copyable handle: shared ownership of accessibility, not of the IR node.
//!
//! ``TensorGraph`` keeps ``unique_ptr<TensorNode>``. Ops store raw
//! ``TensorNode *``. When the last ``TensorRef`` for a node is destroyed,
//! ``tensor::unregister`` is recorded (StarPU handle GC); the node object
//! remains (tech debt D1: TensorNode IR is not destroyed).
class TensorRef
{
  public:
    TensorRef() = default;

    //! Adopt an existing graph-owned node (shares one hold per node).
    static TensorRef adopt(TensorGraph::TensorNode *node);

    TensorGraph::TensorNode *get() const noexcept;
    TensorGraph::TensorNode &operator*() const { return *get(); }
    TensorGraph::TensorNode *operator->() const { return get(); }

    //! Convenience for op APIs that take ``TensorNode *``.
    //! Do not store the result of a temporary ``TensorRef`` in a raw pointer
    //! variable - keep a ``TensorRef`` local instead.
    operator TensorGraph::TensorNode *() const noexcept { return get(); }

    explicit operator bool() const noexcept { return get() != nullptr; }

    bool operator==(TensorRef const &other) const noexcept
    {
        return get() == other.get();
    }
    bool operator!=(TensorRef const &other) const noexcept
    {
        return !(*this == other);
    }

  private:
    struct Hold;
    explicit TensorRef(std::shared_ptr<Hold> hold) : hold_(std::move(hold)) {}
    std::shared_ptr<Hold> hold_;
};

//! True while at least one ``TensorRef`` refers to \p node.
bool tensor_ref_is_live(TensorGraph::TensorNode const *node) noexcept;

//! Gate ``TensorRef`` hold destructors (set false before destroying graphs).
void set_tensor_nodes_alive(bool alive);
bool tensor_nodes_alive() noexcept;

//! Queue logicals whose last ``TensorRef`` died (for compile-time payload flush).
void note_tensor_ref_released(TensorGraph::TensorNode *node);
std::vector<TensorGraph::TensorNode *> take_released_tensor_refs();

} // namespace nntile
