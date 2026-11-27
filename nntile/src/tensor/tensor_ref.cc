/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file nntile/src/tensor/tensor_ref.cc
 * TensorRef hold lifetime -> async unregister.
 *
 * @version 1.1.0
 * */

#include <nntile/tensor/tensor_ref.hh>

#include <atomic>
#include <mutex>
#include <unordered_set>

#include <nntile/tensor/graph_data_node.hh>
#include <nntile/tensor/ops/unregister.hh>

namespace nntile
{

namespace
{
std::atomic<bool> g_tensor_nodes_alive{true};
std::atomic<std::uint64_t> g_tensor_nodes_generation{0};
std::atomic<std::size_t> g_tensor_refs_alive{0};
std::recursive_mutex g_tensor_nodes_destruction_mutex;
std::mutex g_released_mutex;
std::unordered_set<TensorGraph::TensorNode *> g_released;
} // namespace

void set_tensor_nodes_alive(bool alive)
{
    g_tensor_nodes_alive.store(alive, std::memory_order_release);
    if (!alive)
    {
        std::lock_guard<std::mutex> lock(g_released_mutex);
        g_released.clear();
    }
}

bool tensor_nodes_alive() noexcept
{
    return g_tensor_nodes_alive.load(std::memory_order_acquire);
}

std::uint64_t tensor_nodes_generation() noexcept
{
    return g_tensor_nodes_generation.load(std::memory_order_acquire);
}

bool tensor_refs_alive() noexcept
{
    return g_tensor_refs_alive.load(std::memory_order_acquire) != 0;
}

void bump_tensor_nodes_generation() noexcept
{
    g_tensor_nodes_generation.fetch_add(1, std::memory_order_acq_rel);
}

void tensor_nodes_destruction_lock() noexcept
{
    g_tensor_nodes_destruction_mutex.lock();
}

void tensor_nodes_destruction_unlock() noexcept
{
    g_tensor_nodes_destruction_mutex.unlock();
}

void note_tensor_ref_released(TensorGraph::TensorNode *node)
{
    if (node == nullptr || !tensor_nodes_alive())
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_released_mutex);
    g_released.insert(node);
}

std::vector<TensorGraph::TensorNode *> take_released_tensor_refs()
{
    std::lock_guard<std::mutex> lock(g_released_mutex);
    std::vector<TensorGraph::TensorNode *> out;
    out.reserve(g_released.size());
    for (TensorGraph::TensorNode *n : g_released)
    {
        out.push_back(n);
    }
    g_released.clear();
    return out;
}

struct TensorRef::Hold
{
    TensorGraph::TensorNode *node = nullptr;
    std::uint64_t generation = 0;

    explicit Hold(TensorGraph::TensorNode *n):
        node(n), generation(tensor_nodes_generation())
    {
        if (node != nullptr)
        {
            g_tensor_refs_alive.fetch_add(1, std::memory_order_acq_rel);
            node->note_external_hold_();
        }
    }

    ~Hold()
    {
        if (node == nullptr)
        {
            return;
        }
        g_tensor_refs_alive.fetch_sub(1, std::memory_order_acq_rel);
        // Exclude a concurrent graph destruction: the recorder holds this
        // lock across the generation bump and node-storage free, so either
        // we see the old generation and the node is still alive, or we see
        // the new one and never dereference it.
        std::lock_guard<std::recursive_mutex> guard(g_tensor_nodes_destruction_mutex);
        TensorGraph::TensorNode *n = node;
        node = nullptr;
        // The graph that owned n may have been destroyed (reset/shutdown,
        // then possibly a new graph armed the alive flag again). The
        // generation captured at construction is the only safe test: a
        // stale generation means n points into freed storage, so do not
        // even call note_external_release_ through it. A plain alive=false
        // toggle (callers use it to suppress unregister recording) does
        // NOT bump the generation, so the node is still valid and its hold
        // count must still be released.
        if (generation != tensor_nodes_generation())
        {
            return;
        }
        n->note_external_release_();
        if (!tensor_nodes_alive())
        {
            return;
        }
        if (n->graph() != nullptr)
        {
            tensor::unregister(n);
        }
    }

    Hold(Hold const &) = delete;
    Hold &operator=(Hold const &) = delete;
};

TensorRef TensorRef::adopt(TensorGraph::TensorNode *node)
{
    if (node == nullptr)
    {
        return TensorRef{};
    }
    if (auto existing =
            std::static_pointer_cast<Hold>(node->external_hold_.lock()))
    {
        return TensorRef{std::move(existing)};
    }
    auto hold = std::make_shared<Hold>(node);
    node->external_hold_ = hold;
    return TensorRef{std::move(hold)};
}

TensorGraph::TensorNode *TensorRef::get() const noexcept
{
    return hold_ ? hold_->node : nullptr;
}

bool TensorRef::is_current() const noexcept
{
    return hold_ && hold_->node != nullptr
        && hold_->generation == tensor_nodes_generation();
}

bool tensor_ref_is_live(TensorGraph::TensorNode const *node) noexcept
{
    return node != nullptr && node->external_hold_count() > 0;
}

} // namespace nntile
