/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/tile.hh
 * Tile<T> class
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/traits.hh>
#include <nntile/starpu/config.hh>

namespace nntile::tile
{

// Forward declaration
template<typename T>
class TileLocalData;

//! Many-dimensional tensor, stored contiguously in a Fortran order
/*! Underlying StarPU data is variable, as we need only address and size of a
 * contiguous memory.
 * */
template<typename T>
class Tile: public TileTraits, public starpu::VariableHandle
{
    // Check if provided memory is enough to store data
    size_t _get_size(Index ptr_nelems)
    {
        // At first check if provided storage is enough
        if(nelems > ptr_nelems)
        {
            throw std::runtime_error("Required memory size is larger than "
                    "actually allocated memory");
        }
        // Check if total size is within size_t type
        std::size_t size = nelems * sizeof(T);
        if(size / sizeof(T) != nelems)
        {
            throw std::runtime_error("Type size_t is not enough to hold size "
                    "of provided buffer");
        }
        return size;
    }
public:
    //! Construct a tile from traits and StarPU handle
    Tile(const TileTraits &traits_, const starpu::VariableHandle &handle_):
        TileTraits(traits_),
        starpu::VariableHandle(handle_)
    {
    }
    //! Construct a tile, allocated/deallocated by StarPU
    explicit Tile(const std::vector<Index> &shape_):
        TileTraits(shape_),
        starpu::VariableHandle(nelems*sizeof(T), STARPU_R)
    {
    }
    //! Construct a tile, allocated/deallocated by StarPU
    explicit Tile(const TileTraits &traits):
        TileTraits(traits),
        starpu::VariableHandle(nelems*sizeof(T), STARPU_R)
    {
    }
    //! Construct a tile out of provided contiguous memory buffer
    Tile(const std::vector<Index> &shape_, T *ptr, Index ptr_nelems):
        TileTraits(shape_),
        starpu::VariableHandle(ptr, _get_size(ptr_nelems), STARPU_RW)
    {
    }
    //! Construct a tile out of provided contiguous memory buffer
    Tile(const TileTraits &traits, T *ptr, Index ptr_nelems):
        TileTraits(traits),
        starpu::VariableHandle(ptr, _get_size(ptr_nelems), STARPU_RW)
    {
    }
    TileLocalData<T> acquire(starpu_data_access_mode mode)
        const;
};

//! Local copy of a tile in CPU RAM
/*! This is an auxiliary class for debugging and testing
 * */
template<typename T>
class TileLocalData: public starpu::HandleLocalData
{
public:
    TileLocalData(const Tile<T> &tile, starpu_data_access_mode mode):
        starpu::HandleLocalData(tile, mode)
    {
    }
    const T &operator[](Index i)
        const
    {
        return get_ptr()[i];
    }
    T &operator[](Index i)
    {
        return get_ptr()[i];
    }
    T *get_ptr() const
    {
        return reinterpret_cast<T *>(starpu::HandleLocalData::get_ptr());
    }
};

//! Acquire tile locally in CPU RAM
template<typename T>
TileLocalData<T> Tile<T>::acquire(starpu_data_access_mode mode)
    const
{
    return TileLocalData<T>(*this, mode);
}

} // namespace nntile::tile
