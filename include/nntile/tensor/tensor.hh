/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/tensor.hh
 * Tensor<T> class
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/traits.hh>
#include <nntile/tile/tile.hh>

namespace nntile
{

//! Many-dimensional tensor, presented by a set of subtensors (tiles)
//
// This is the main data storage class, that assumes a tensor as a set of
// tiles, handled by StarPU runtime system.
template<typename T>
class Tensor: public TensorTraits
{
public:
    //! Pointer to the contiguous memory
    //std::shared_ptr<void> ptr;
    //! Total size of allocated memory in bytes
    Index alloc_size;
    //! Tiles
    std::vector<Tile<T>> tiles;
    //! Constructor
    Tensor(const TensorTraits &traits,
            Index alignment=16):
        TensorTraits(traits),
        //ptr(),
        alloc_size(0),
        tiles()
    {
        // Check if alignment is positive
        if(alignment <= 0)
        {
            throw std::runtime_error("alignment <= 0");
        }
        // At first compute memory footprint and offsets for each tile
        std::vector<Index> tiles_nelems(grid.nelems);
        std::vector<Index> tiles_offset(grid.nelems);
        for(Index i = 0; i < grid.nelems; ++i)
        {
            // Remember offset to current tile
            tiles_offset[i] = alloc_size;
            // Get tile index
            const auto tile_index = grid.linear_to_index(i);
            // Get shape of corresponding tile
            const auto tile_shape = TensorTraits::get_tile_shape(tile_index);
            // Actual memory for the tile in elements T
            tiles_nelems[i] = 1;
            for(Index j = 0; j < ndim; ++j)
            {
                tiles_nelems[i] *= tile_shape[j];
            }
            // Total memory for tile in bytes
            Index tile_alloc_size = tiles_nelems[i] * sizeof(T);
            // Compute offset only if allocation is non-zero
            if(tile_alloc_size != 0)
            {
                // Round up to the alignment parameter
                Index naligns = (tile_alloc_size-1)/alignment + 1;
                // Update allocation size
                alloc_size += naligns * alignment;
            }
        }
        // Allocate memory
        //void *ptr_raw;
        //int ret = starpu_malloc(&ptr_raw, alloc_size);
        //if(ret != 0)
        //{
        //    throw std::runtime_error("ret != 0");
        //}
        //char *ptr_char = reinterpret_cast<char *>(ptr_raw);
        //ptr = std::shared_ptr<void>(ptr_raw, starpu_free);
        // Register tiles
        tiles.reserve(grid.nelems);
        for(Index i = 0; i < grid.nelems; ++i)
        {
            // Get tile index
            const auto tile_index = grid.linear_to_index(i);
            // Get shape of corresponding tile
            const auto tile_shape = TensorTraits::get_tile_shape(tile_index);
            //tiles.emplace_back(tile_shape,
            //        reinterpret_cast<T *>(&ptr_char[tiles_offset[i]]),
            //        tiles_nelems[i]);
            tiles.emplace_back(tile_shape);
        }
    }
    //! Constructor
    Tensor(const std::vector<Index> &shape_,
            const std::vector<Index> &basetile_shape_,
            Index alignment=16):
        Tensor({shape_, basetile_shape_}, alignment)
    {
    }
    const Tile<T> &get_tile(Index linear_offset) const
    {
        if(linear_offset < 0 or linear_offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tiles[linear_offset];
    }
    const Tile<T> &get_tile(const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tiles[linear_offset];
    }
    const TileTraits &get_tile_traits(Index linear_offset) const
    {
        return get_tile(linear_offset);
    }
    const TileTraits &get_tile_traits(const std::vector<Index> &tile_index)
        const
    {
        return get_tile(tile_index);
    }
    //! Unregister underlying handles without waiting for destructor
    void unregister()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            tiles[i].unregister();
        }
    }
};

} // namespace nntile

