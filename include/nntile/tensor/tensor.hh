#pragma once

#include <nntile/tensor/traits.hh>
#include <nntile/tile/tile.hh>
#include <memory>

namespace nntile
{

template<typename T>
class Tensor: public TensorTraits
{
public:
    //! Pointer to the contiguous memory
    std::shared_ptr<std::byte[]> ptr;
    //! Total size of allocated memory in bytes
    size_t alloc_size;
    //! Tiles
    std::vector<Tile<T>> tiles;
    //! Constructor
    Tensor(const TensorTraits &traits,
            size_t alignment=16):
        TensorTraits(traits),
        ptr(),
        alloc_size(0),
        tiles()
    {
        // At first compute memory footprint and offsets for each tile
        std::vector<size_t> tiles_nelems(grid.nelems);
        std::vector<size_t> tiles_offset(grid.nelems);
        for(size_t i = 0; i < grid.nelems; ++i)
        {
            // Remember offset to current tile
            tiles_offset[i] = alloc_size;
            // Get shape of corresponding tile
            const auto tile_shape = TensorTraits::get_tile_shape(i);
            // Actual memory for the tile in elements T
            tiles_nelems[i] = 1;
            for(size_t j = 0; j < ndim; ++j)
            {
                tiles_nelems[i] *= tile_shape[j];
            }
            // Total memory for tile in bytes
            size_t tile_alloc_size = tiles_nelems[i] * sizeof(T);
            // Compute offset only if allocation is non-zero
            if(tile_alloc_size != 0)
            {
                // Round up to the alignment parameter
                size_t naligns = (tile_alloc_size-1)/alignment + 1;
                // Update allocation size
                alloc_size += naligns * alignment;
            }
        }
        // Allocate memory
        auto ptr_raw = ::new std::byte[alloc_size];
        ptr = std::shared_ptr<std::byte[]>(ptr_raw);
        // Register tiles
        tiles.reserve(grid.nelems);
        for(size_t i = 0; i < grid.nelems; ++i)
        {
            tiles.emplace_back(TensorTraits::get_tile_shape(i),
                    reinterpret_cast<T *>(&ptr_raw[tiles_offset[i]]),
                    tiles_nelems[i]);
        }
    }
    //! Constructor
    Tensor(const std::vector<size_t> &shape_,
            const std::vector<size_t> &basetile_shape_,
            size_t alignment=16):
        Tensor({shape_, basetile_shape_}, alignment)
    {
    }
    const Tile<T> &get_tile(size_t offset) const
    {
        if(offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tiles[offset];
    }
    const Tile<T> &get_tile(const std::vector<size_t> &index) const
    {
        size_t offset = get_tile_offset(index);
        return tiles[offset];
    }
    const TileTraits &get_tile_traits(size_t offset) const
    {
        if(offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tiles[offset];
    }
    const TileTraits &get_tile_traits(const std::vector<size_t> &index) const
    {
        size_t offset = get_tile_offset(index);
        return tiles[offset];
    }
    const std::vector<size_t> &get_tile_shape(size_t offset) const
    {
        if(offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tiles[offset].shape;
    }
    const std::vector<size_t> &get_tile_shape(
            const std::vector<size_t> &index) const
    {
        size_t offset = get_tile_offset(index);
        return tiles[offset].shape;
    }
};

} // namespace nntile

