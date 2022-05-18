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

// Forward declaration
template<typename T>
class TileLocalData;

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
    TileLocalData<T> acquire(enum starpu_data_access_mode mode)
        const;
};

template<typename T>
class TileLocalData: public StarpuHandleLocalData
{
public:
    TileLocalData(const Tile<T> &tile, enum starpu_data_access_mode mode):
        StarpuHandleLocalData(tile, mode)
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
        return reinterpret_cast<T *>(StarpuHandleLocalData::get_ptr());
    }
};

template<typename T>
TileLocalData<T> Tile<T>::acquire(enum starpu_data_access_mode mode)
    const
{
    return TileLocalData<T>(*this, mode);
}

} // namespace nntile

