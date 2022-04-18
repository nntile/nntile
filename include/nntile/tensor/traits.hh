#pragma once

#include <nntile/tile/traits.hh>

namespace nntile
{

class TensorTraits: public TileTraits
{
    const std::vector<size_t> &_get_basetile_shape(
            const std::vector<size_t> &basetile_shape) const
    {
        // Check dimension of base tile
        if(basetile_shape.size() != ndim)
        {
            throw std::runtime_error("basetile_shape.size() != ndim");
        }
        // Check if base tile has zero entry somewhere
        for(size_t i = 0; i < ndim; ++i)
        {
            if(basetile_shape[i] == 0 and shape[i] != 0)
            {
                throw std::runtime_error("basetile_shape[i] == 0 and "
                        "shape[i] != 0");
            }
        }
        return basetile_shape;
    }
    std::vector<size_t> _get_grid_shape() const
    {
        // Define grid of tiles
        std::vector<size_t> grid_shape(ndim, 0);
        for(size_t i = 0; i < ndim; ++i)
        {
            if(shape[i] != 0)
            {
                // Round up number of tiles
                size_t tmp = (shape[i]-1)/basetile_shape[i] + 1;
                grid_shape[i] = tmp;
            }
        }
        return grid_shape;
    }
    std::vector<size_t> _get_leftover_shape() const
    {
        // Define leftover size in each dimension
        std::vector<size_t> leftover_shape(ndim, 0);
        for(size_t i = 0; i < ndim; ++i)
        {
            if(shape[i] != 0)
            {
                // Simply get size of the last tile
                leftover_shape[i] = shape[i] -
                    (grid.shape[i]-1)*basetile_shape[i];
            }
        }
        return leftover_shape;
    }
public:
    //! Shape of base tile
    std::vector<size_t> basetile_shape;
    //! Grid of tiles viewed as tile of tiles
    TileTraits grid;
    //! Leftover size in each dimension
    std::vector<size_t> leftover_shape;
    //! Constructor
    TensorTraits(const std::vector<size_t> &shape_,
            const std::vector<size_t> &basetile_shape_):
        TileTraits(shape_),
        basetile_shape(_get_basetile_shape(basetile_shape_)),
        grid(_get_grid_shape()),
        leftover_shape(_get_leftover_shape())
    {
    }
    std::vector<size_t> get_tile_index(size_t offset) const
    {
        // Check if actual tile exists
        if(offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        if(ndim == 0)
        {
            return std::vector<size_t>();
        }
        std::vector<size_t> index(ndim);
        // Avoid i=0-1 in size_t type
        for(size_t i = ndim-1; i >= 1; --i)
        {
            const size_t div = offset / grid.stride[i];
            offset -= div * grid.stride[i];
            index[i] = div;
        }
        index[0] = offset;
        return index;
    }
    size_t get_tile_offset(const std::vector<size_t> &index) const
    {
        // Check if index is in correct range
        if(index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        if(ndim == 0)
        {
            return 0;
        }
        if(index[0] >= grid.shape[0])
        {
            throw std::runtime_error("Grid index out of bounds");
        }
        size_t offset = index[0]; // grid.stride[0] = 1
        for(size_t i = 1; i < ndim; ++i)
        {
            // Check if index is in correct range
            if(index[i] >= grid.shape[i])
            {
                throw std::runtime_error("Grid index out of bounds");
            }
            offset += index[i] * grid.stride[i];
        }
        return offset;
    }
    std::vector<size_t> get_tile_shape(size_t offset) const
    {
        // Get index of tile and init its shape with base tile shape,
        // offset is checked in get_tile_index() function
        std::vector<size_t> shape(basetile_shape),
            index(get_tile_index(offset));
        // Update tile shape if its index is last in corresponding dimension
        for(size_t i = 0; i < ndim; ++i)
        {
            if(index[i]+1 == grid.shape[i])
            {
                shape[i] = leftover_shape[i];
            }
        }
        return shape;
    }
    std::vector<size_t> get_tile_shape(const std::vector<size_t> &index) const
    {
        // Init tile shape with base tile shape
        std::vector<size_t> shape(basetile_shape);
        // Update tile shape if necessary
        for(size_t i = 0; i < ndim; ++i)
        {
            // Check if index is actually within bounds
            if(index[i] >= grid.shape[i])
            {
                throw std::runtime_error("index is out of bounds");
            }
            // If tile is the last in corresponding dimension
            else if(grid.shape[i]-index[i] == 1)
            {
                shape[i] = leftover_shape[i];
            }
        }
        return shape;
    }
    friend std::ostream &operator<<(std::ostream &os,
            const TensorTraits &traits);
};

} // namespace nntile

