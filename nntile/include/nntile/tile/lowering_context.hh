/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/lowering_context.hh
 * Context for TensorGraph::OpNode::lower_to_tile.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

// NNTile headers
#include <nntile/tile/graph_decl.hh>
#include <nntile/tile/graph_data_node.hh>

namespace nntile
{

//! Dense TensorNode.id()-indexed map to tile lists (O(1) lookup).
//!
//! Replaces ``std::map<TensorNode*, vector<TileNode*>>`` so lowering stays
//! O(M) rather than O(M log T). Empty / erased slots are holes (allowed
//! after GC); occupancy is tracked explicitly.
class TensorNodeToTileMap
{
  public:
    using key_type = TensorGraph::TensorNode const *;
    using mapped_type = std::vector<TileGraph::TileNode *>;

    mapped_type &operator[](key_type node)
    {
        Slot &slot = ensure_slot(node);
        slot.occupied = true;
        slot.key = node;
        return slot.tiles;
    }

    mapped_type *try_get(key_type node)
    {
        if (node == nullptr)
        {
            return nullptr;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size() || !slots_[id].occupied)
        {
            return nullptr;
        }
        return &slots_[id].tiles;
    }

    mapped_type const *try_get(key_type node) const
    {
        if (node == nullptr)
        {
            return nullptr;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size() || !slots_[id].occupied)
        {
            return nullptr;
        }
        return &slots_[id].tiles;
    }

    bool contains(key_type node) const
    {
        return try_get(node) != nullptr;
    }

    size_t count(key_type node) const
    {
        return contains(node) ? 1 : 0;
    }

    void erase(key_type node)
    {
        if (node == nullptr)
        {
            return;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size())
        {
            return;
        }
        slots_[id].occupied = false;
        slots_[id].key = nullptr;
        slots_[id].tiles.clear();
        slots_[id].tiles.shrink_to_fit();
    }

    void clear()
    {
        slots_.clear();
    }

  private:
    struct Slot
    {
        bool occupied = false;
        key_type key = nullptr;
        mapped_type tiles;
    };

    Slot &ensure_slot(key_type node)
    {
        if (node == nullptr)
        {
            throw std::invalid_argument(
                "TensorNodeToTileMap: null TensorNode");
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size())
        {
            slots_.resize(id + 1);
        }
        return slots_[id];
    }

    std::vector<Slot> slots_;
};

//! Dense map from TensorNode.id() to a value (O(1) lookup).
template <typename T>
class TensorNodeIdMap
{
  public:
    using key_type = TensorGraph::TensorNode const *;

    T &operator[](key_type node)
    {
        Slot &slot = ensure_slot(node);
        slot.occupied = true;
        return slot.value;
    }

    T *try_get(key_type node)
    {
        if (node == nullptr)
        {
            return nullptr;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size() || !slots_[id].occupied)
        {
            return nullptr;
        }
        return &slots_[id].value;
    }

    T const *try_get(key_type node) const
    {
        if (node == nullptr)
        {
            return nullptr;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size() || !slots_[id].occupied)
        {
            return nullptr;
        }
        return &slots_[id].value;
    }

    bool contains(key_type node) const
    {
        return try_get(node) != nullptr;
    }

    size_t count(key_type node) const
    {
        return contains(node) ? 1 : 0;
    }

    void erase(key_type node)
    {
        if (node == nullptr)
        {
            return;
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size())
        {
            return;
        }
        slots_[id].occupied = false;
        slots_[id].value = T{};
    }

    void clear()
    {
        slots_.clear();
    }

  private:
    struct Slot
    {
        bool occupied = false;
        T value{};
    };

    Slot &ensure_slot(key_type node)
    {
        if (node == nullptr)
        {
            throw std::invalid_argument("TensorNodeIdMap: null TensorNode");
        }
        auto const id = static_cast<size_t>(node->id());
        if (id >= slots_.size())
        {
            slots_.resize(id + 1);
        }
        return slots_[id];
    }

    std::vector<Slot> slots_;
};

//! Passed to each tensor op when lowering to TileGraph.
struct LoweringContext
{
    TileGraph &out;
    TensorNodeToTileMap const &tile_map;
    TensorGraphTiling const &tiling;
};

} // namespace nntile
