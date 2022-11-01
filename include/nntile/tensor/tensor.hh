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
 * @date 2022-09-27
 * */

#pragma once

#include <nntile/tensor/traits.hh>
#include <nntile/tile/tile.hh>
#include <starpu_mpi.h>

namespace nntile
{
namespace tensor
{

//! Many-dimensional tensor, presented by a set of subtensors (tiles)
//
// This is the main data storage class, that assumes a tensor as a set of
// tiles, handled by StarPU runtime system.
template<typename T>
class Tensor: public TensorTraits
{
public:
    //! Traits of all tiles
    std::vector<tile::TileTraits> tile_traits;
    //! StarPU handles of all tiles
    std::vector<starpu::VariableHandle> tile_handles;
    //! Constructor
    explicit Tensor(const TensorTraits &traits,
            const std::vector<int> &distribution,
            starpu_mpi_tag_t &last_tag):
        TensorTraits(traits)
    {
        // Check distribution
        if(distribution.size() != grid.nelems)
        {
            throw std::runtime_error("Wrong distribution");
        }
        // Register tiles
        tile_traits.reserve(grid.nelems);
        tile_handles.reserve(grid.nelems);
        for(Index i = 0; i < grid.nelems; ++i)
        {
            // At first check if last tag is less than maximal tag
            // Get tile index
            const auto tile_index = grid.linear_to_index(i);
            // Get shape of corresponding tile
            const auto tile_shape = TensorTraits::get_tile_shape(tile_index);
            // Generate traits for the tile
            tile_traits.emplace_back(tile_shape);
            // Set StarPU-managed handle
            tile_handles.emplace_back(sizeof(T)*tile_traits[i].nelems,
                    STARPU_R);
            // Register tile with MPI
            starpu_mpi_data_register(
                    static_cast<starpu_data_handle_t>(tile_handles[i]),
                    last_tag, distribution[i]);
            ++last_tag;
        }
    }
    tile::Tile<T> get_tile(Index linear_offset) const
    {
        if(linear_offset < 0 or linear_offset >= grid.nelems)
        {
            throw std::runtime_error("Tile offset is out of bounds");
        }
        return tile::Tile<T>(tile_traits[linear_offset],
                tile_handles[linear_offset]);
    }
    tile::Tile<T> get_tile(const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile::Tile<T>(tile_traits[linear_offset],
                tile_handles[linear_offset]);
    }
    const tile::TileTraits &get_tile_traits(Index linear_offset) const
    {
        return tile_traits[linear_offset];
    }
    const tile::TileTraits &get_tile_traits(
            const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile_traits[linear_offset];
    }
    const starpu::Handle &get_tile_handle(Index linear_offset) const
    {
        return tile_handles[linear_offset];
    }
    const starpu::Handle &get_tile_handle(
            const std::vector<Index> &tile_index) const
    {
        Index linear_offset = grid.index_to_linear(tile_index);
        return tile_handles[linear_offset];
    }
    //! Unregister underlying handles without waiting for destructor
    void unregister()
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            tile_handles[i].unregister();
        }
    }
    //! Invalidate tensor values
    void invalidate_submit() const
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = static_cast<starpu_data_handle_t>(get_tile_handle(i));
            starpu_data_invalidate_submit(tmp);
        }
    }
    //! Advice to evict data from GPU
    void wont_use() const
    {
        for(Index i = 0; i < grid.nelems; ++i)
        {
            auto tmp = static_cast<starpu_data_handle_t>(get_tile_handle(i));
            starpu_data_wont_use(tmp);
        }
    }
};

} // namespace tensor
} // namespace nntile

