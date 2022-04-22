/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/tile.hh
 * Tile<T> class
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/traits.hh>
#include <nntile/starpu.hh>

namespace nntile
{

//! Many-dimensional tensor, stored contiguously in a Fortran order
//
// This is the main data storage class, that is handled by StarPU runtime
// system.
template<typename T>
class Tile: public TileTraits, public StarpuVariableHandle
{
    size_t check_nelems(Index req_nelems, Index ptr_nelems)
    {
        if(req_nelems > ptr_nelems)
        {
            throw std::runtime_error("Required memory size is larger than "
                    "actually allocated memory");
        }
        size_t result = req_nelems;
        if(req_nelems != result)
        {
            throw std::runtime_error("req_nelems != result");
        }
        return result;
    }
public:
    Tile(const std::vector<Index> &shape_):
        TileTraits(shape_),
        StarpuVariableHandle(nelems*sizeof(T))
    {
    }
    Tile(const TileTraits &traits):
        TileTraits(traits),
        StarpuVariableHandle(nelems*sizeof(T))
    {
    }
    Tile(const std::vector<Index> &shape_, T *ptr, Index ptr_nelems):
        TileTraits(shape_),
        StarpuVariableHandle(reinterpret_cast<uintptr_t>(ptr),
                sizeof(T)*check_nelems(nelems, ptr_nelems))
    {
    }
    Tile(const TileTraits &traits, T *ptr, Index ptr_nelems):
        TileTraits(traits),
        StarpuVariableHandle(reinterpret_cast<uintptr_t>(ptr),
                sizeof(T)*check_nelems(nelems, ptr_nelems))
    {
    }
    //! Get pointer to local data if corresponding interface supports it
    const T *get_local_ptr() const
    {
        return reinterpret_cast<const T *>(StarpuHandle::get_local_ptr());
    }
    //! Read an element of the tile by its coordinate
    T at_index(const std::vector<Index> &index) const
    {
        // Get the corresponding tile data to the local buffer
        acquire(STARPU_R);
        // Read the value, bounds are checked in index_to_linear() function
        T value = get_local_ptr()[index_to_linear(index)];
        // Release the buffer
        release();
        return value;
    }
    //! Read an element of the tile by its linear memory offset
    T at_linear(Index linear_offset) const
    {
        // Check bounds
        if(linear_offset < 0 or linear_offset >= nelems)
        {
            throw std::runtime_error("Index out of bounds");
        }
        // Get the corresponding tile data to the local buffer
        acquire(STARPU_R);
        // Read the value
        T value = get_local_ptr()[linear_offset];
        // Release the buffer
        release();
        return value;
    }
};

} // namespace nntile

