/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/traits.hh
 * Integer properties of the Tensor<T> class
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/traits.hh>

namespace nntile::tensor
{

//! Integer arithmetics for tensors
class TensorTraits: public tile::TileTraits
{
    static const std::vector<Index> &_get_basetile_shape(Index ndim,
            const std::vector<Index> &basetile_shape)
    {
        // Check dimension of base tile
        if(basetile_shape.size() != ndim)
        {
            throw std::runtime_error("basetile_shape.size() != ndim");
        }
        // Check if base tile has only positive values
        for(Index i = 0; i < ndim; ++i)
        {
            if(basetile_shape[i] <= 0)
            {
                throw std::runtime_error("basetile_shape[i] <= 0");
            }
        }
        return basetile_shape;
    }
    static std::vector<Index> _get_grid_shape(Index ndim,
            const std::vector<Index> &shape,
            const std::vector<Index> &basetile_shape)
    {
        // Define grid of tiles
        std::vector<Index> grid_shape(ndim);
        for(Index i = 0; i < ndim; ++i)
        {
            // Round up number of tiles
            grid_shape[i] = (shape[i]-1)/basetile_shape[i] + 1;
        }
        return grid_shape;
    }
    static std::vector<Index> _get_leftover_shape(Index ndim,
            const std::vector<Index> &shape,
            const std::vector<Index> &basetile_shape,
            const std::vector<Index> &grid_shape)
    {
        // Define leftover size in each dimension
        std::vector<Index> leftover_shape(ndim, 0);
        for(Index i = 0; i < ndim; ++i)
        {
            // Simply get size of the last tile
            leftover_shape[i] = shape[i] - (grid_shape[i]-1)*basetile_shape[i];
        }
        return leftover_shape;
    }
public:
    //! Shape of base tile
    std::vector<Index> basetile_shape;
    //! Grid of tiles viewed as tile of tiles
    tile::TileTraits grid;
    //! Leftover size in each dimension
    std::vector<Index> leftover_shape;
    //! Constructor
    explicit TensorTraits(const std::vector<Index> &shape_,
            const std::vector<Index> &basetile_shape_):
        tile::TileTraits(shape_),
        basetile_shape(_get_basetile_shape(ndim, basetile_shape_)),
        grid(_get_grid_shape(ndim, shape, basetile_shape_)),
        leftover_shape(_get_leftover_shape(ndim, shape, basetile_shape,
                    grid.shape))
    {
    }
    //! Get shape of a tile at given coordinate of grid of tiles
    std::vector<Index> get_tile_shape(const std::vector<Index> &tile_index)
        const
    {
        // Check number of dimensions
        if(tile_index.size() != ndim)
        {
            throw std::runtime_error("Wrong dimensionality");
        }
        // Init tile shape with base tile shape
        std::vector<Index> tile_shape(basetile_shape);
        // Update tile shape if necessary
        for(Index i = 0; i < ndim; ++i)
        {
            // Check if index is actually within bounds
            if(tile_index[i] < 0 or tile_index[i] >= grid.shape[i])
            {
                throw std::runtime_error("Index out of bounds");
            }
            // If tile is the last in corresponding dimension
            else if(grid.shape[i]-tile_index[i] == 1)
            {
                tile_shape[i] = leftover_shape[i];
            }
        }
        return tile_shape;
    }
    friend std::ostream &operator<<(std::ostream &os,
            const TensorTraits &traits);
};

} // namespace nntile::tensor
