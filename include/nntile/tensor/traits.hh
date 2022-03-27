#pragma once

#include <nntile/tile.hh>

namespace nntile
{

struct TensorTraits
{
    //! Run-time dimensionality
    int ndim;
    //! Shape of tensor
    std::vector<int> shape;
    //! Shape of base tile
    std::vector<int> tile_shape;
    //! Leftover size in each dimension
    std::vector<int> leftover_shape;
    //! Shape of grid of tiles
    std::vector<int> grid_shape;
    //! Stride of grid for fast indexing of tiles
    std::vector<int> grid_stride;
    //! Number of tiles in grid
    int ntiles;
    //! Shapes of all possible reshapes of tensor grid into matrix grids
    //
    // grid_matrix_shape[0] is a (prod(grid_shape[0:0]),
    // prod(grid_shape[1:ndim-1]) reshape grid_matrix_shape[1] is a
    // (prod(grid_shape[0:1]), prod(grid_shape[2:ndim-1]) reshape
    // and so on, grid_matrix_shape[ndim-2] is a (prod(sgrid_hape[0:ndim-2]),
    // prod(grid_shape[ndim-1:ndim-1]) reshape
    std::vector<std::array<int, 2>> grid_matrix_shape;
    //! Index of each tile in grid
    std::vector<std::vector<int>> tiles_index;
    //! Collection of traits of every tile in grid
    std::vector<TileTraits> tiles_traits;
    //! Constructor
    TensorTraits(const std::vector<int> &shape_,
            const std::vector<int> &tile_shape_):
        ndim(shape_.size()),
        shape(shape_),
        tile_shape(tile_shape_),
        leftover_shape(ndim),
        grid_shape(ndim),
        grid_stride(ndim),
        grid_matrix_shape(ndim-1)
    {
        // Check dimensions
        if(ndim == 0)
        {
            throw std::runtime_error("shape must be non-empty");
        }
        // Double-check conversion from size_t to int
        if(ndim != shape.size())
        {
            throw std::runtime_error("size_t to int conversion overflow");
        }
        // Check dimension of tile
        if(tile_shape.size() != ndim)
        {
            throw std::runtime_error("tile_shape.size() != ndim");
        }
        // Check if input shape is positive
        for(int i = 0; i < ndim; ++i)
        {
            if(shape[i] <= 0)
            {
                throw std::runtime_error("shape must be positive");
            }
            if(tile_shape[i] <= 0)
            {
                throw std::runtime_error("tile_shape must be positive");
            }
        }
        // Define grid of tiles
        for(int i = 0; i < ndim; ++i)
        {
            int tmp = (shape[i]-1)/tile_shape[i] + 1;
            grid_shape[i] = tmp;
            leftover_shape[i] = shape[i] - (grid_shape[i]-1)*tile_shape[i];
        }
        // Define properties of grid
        size_t tmp_long = grid_shape[0];
        grid_matrix_shape[0][0] = grid_shape[0];
        grid_matrix_shape[ndim-2][1] = grid_shape[ndim-1];
        for(int i = 1; i < ndim-1; ++i)
        {
            tmp_long *= grid_shape[i];
            grid_matrix_shape[i][0] = tmp_long;
            grid_matrix_shape[ndim-2-i][1] = grid_matrix_shape[ndim-1-i][1]
                * grid_shape[ndim-1-i];
        }
        tmp_long *= grid_shape[ndim-1];
        ntiles = tmp_long;
        // Check for integer overflow
        if(ntiles != tmp_long)
        {
            throw std::runtime_error("Tile shape is too small or tensor "
                    "shape is too big");
        }
        // Get grid stride
        grid_stride[0] = 1;
        for(int i = 0; i < ndim-1; ++i)
        {
            grid_stride[i+1] = grid_matrix_shape[i][0];
        }
        // Prepare traits of all tiles
        tiles_index.reserve(ntiles);
        tiles_traits.reserve(ntiles);
        // Prepare index
        std::vector<int> current_index(ndim);
        current_index[0] = -1;
        for(int i = 1; i < ndim; ++i)
        {
            current_index[i] = 0;
        }
        // Set traits for each tile
        for(int i = 0; i < ntiles; ++i)
        {
            // Update index of current tile
            ++current_index[0];
            int j = 0;
            while(current_index[j] == grid_shape[j])
            {
                current_index[j] = 0;
                ++j;
                ++current_index[j];
            }
            // Get shape of current tile
            std::vector<int> current_shape(tile_shape);
            for(int j = 0; j < ndim; ++j)
            {
                if(current_index[j] == grid_shape[j]-1)
                {
                    current_shape[j] = leftover_shape[j];
                }
            }
            // Set traits and index of current tile
            tiles_traits.emplace_back(current_shape);
            tiles_index.emplace_back(current_index);
        }
    }
    int offset(const std::vector<int> &grid_index) const
    {
        if(grid_index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        if((grid_index[0] < 0) or (grid_index[0] >= grid_shape[0]))
        {
            throw std::runtime_error("Grid index out of bounds");
        }
        int offset = grid_index[0]; // grid_stride[0]=1
        for(int i = 1; i < ndim; ++i)
        {
            if((grid_index[i] < 0) or (grid_index[i] >= grid_shape[i]))
            {
                throw std::runtime_error("Grid index out of bounds");
            }
            offset += grid_index[i] * grid_stride[i];
        }
        return offset;
    }
    //const std::vector<int> &index(int offset) const
    //{
    //    return tiles_index.at(offset);
    //}
    friend std::ostream &operator<<(std::ostream &os,
            const TensorTraits &traits);
};

std::ostream &operator<<(std::ostream &os, const TensorTraits &traits)
{
    os << "TensorTraits object at " << &traits << "\n";
    os << "shape=(" << traits.shape[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.shape[i];
    }
    os << ")\n";
    os << "tile_shape=(" << traits.tile_shape[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.tile_shape[i];
    }
    os << ")\n";
    os << "leftover_shape=(" << traits.leftover_shape[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.leftover_shape[i];
    }
    os << ")\n";
    os << "grid_shape=(" << traits.grid_shape[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.grid_shape[i];
    }
    os << ")\n";
    os << "grid_stride=(" << traits.grid_stride[0];
    for(int i = 1; i < traits.ndim; ++i)
    {
        os << "," << traits.grid_stride[i];
    }
    os << ")\n";
    os << "ntiles=" << traits.ntiles << "\n";
    os << "grid_matrix_shape=((" << traits.grid_matrix_shape[0][0] <<
        "," << traits.grid_matrix_shape[0][1] << ")";
    for(int i = 1; i < traits.ndim-1; ++i)
    {
        os << ",(" << traits.grid_matrix_shape[i][0] << "," <<
            traits.grid_matrix_shape[i][1] << ")";
    }
    os << ")\n";
    os << "Tiles\n";
    for(int i = 0; i < traits.ntiles; ++i)
    {
        os << "  " << i << "\n";
        os << "    index=(" << traits.tiles_index[i][0];
        for(int j = 1; j < traits.ndim; ++j)
        {
            os << "," << traits.tiles_index[i][j];
        }
        os << ")\n";
        os << "    shape=(" << traits.tiles_traits[i].shape[0];
        for(int j = 1; j < traits.ndim; ++j)
        {
            os << "," << traits.tiles_traits[i].shape[j];
        }
        os << ")\n";
    }
    return os;
}

} // namespace nntile

