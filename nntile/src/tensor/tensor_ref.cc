/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor/tensor_ref.cc
 * TensorRef hold lifetime -> async invalidate.
 *
 * @version 1.1.0
 * */

#include <nntile/tensor/tensor_ref.hh>

#include <atomic>
#include <mutex>
#include <unordered_set>

#include <nntile/tensor/graph_data_node.hh>
#include <nntile/tensor/ops/invalidate.hh>

namespace nntile
{

namespace
{
std::atomic<bool> g_tensor_nodes_alive{true};
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

    explicit Hold(TensorGraph::TensorNode *n) : node(n)
    {
        if (node != nullptr)
        {
            node->note_external_hold_();
        }
    }

    ~Hold()
    {
        if (node == nullptr)
        {
            return;
        }
        TensorGraph::TensorNode *n = node;
        node = nullptr;
        n->note_external_release_();
        if (!tensor_nodes_alive())
        {
            return;
        }
        note_tensor_ref_released(n);
        if (n->graph() != nullptr)
        {
            tensor::invalidate(n);
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

bool tensor_ref_is_live(TensorGraph::TensorNode const *node) noexcept
{
    return node != nullptr && node->external_hold_count() > 0;
}

} // namespace nntile
