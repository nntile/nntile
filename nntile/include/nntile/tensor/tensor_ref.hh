/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/tensor/tensor_ref.hh
 * External accessibility handle for a TensorGraph::TensorNode.
 * Last drop records async unregister; the IR node stays graph-owned.
 *
 * @version 2.0.0
 * */

#pragma once

#include <cstdint>
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
//! ``tensor::unregister`` is recorded (NNHaul handle GC). After
//! ``wait()`` + ``drop_all_ops()``, unreachable IR is destroyed (holes
//! remain in ``data_``).
class TensorRef
{
  public:
    TensorRef() = default;

    //! Adopt an existing graph-owned node (shares one hold per node).
    static TensorRef adopt(TensorGraph::TensorNode *node);

    TensorGraph::TensorNode *get() const noexcept;

    //! False once the graph that owned this node has been destroyed.
    //! Consumers that would dereference get() must reject a stale ref
    //! instead of reading freed storage.
    bool is_current() const noexcept;
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

//! Generation of the live node set. Bumped whenever node storage is
//! destroyed; a Hold captures it so a later graph reset followed by a new
//! graph cannot make a stale raw node pointer look live again.
std::uint64_t tensor_nodes_generation() noexcept;

//! True while any TensorRef Hold exists anywhere. A graph with live
//! holders is retired rather than freed on reset, so a check-then-deref
//! in another thread cannot hit freed node storage.
bool tensor_refs_alive() noexcept;

//! Invalidate every Hold captured so far. Called immediately before the
//! owning graph's node storage is destroyed (not on a plain alive toggle,
//! which some callers use only to suppress unregister recording).
void bump_tensor_nodes_generation() noexcept;

//! Serialize node destruction against off-thread Hold destructors.
//! Holders take the same lock while they check the generation and release
//! the node; the recorder holds it across the generation bump and the
//! graph's node-storage destruction.
void tensor_nodes_destruction_lock() noexcept;
void tensor_nodes_destruction_unlock() noexcept;

//! Queue logicals whose last ``TensorRef`` died (for compile-time payload flush).
void note_tensor_ref_released(TensorGraph::TensorNode *node);
std::vector<TensorGraph::TensorNode *> take_released_tensor_refs();

} // namespace nntile
