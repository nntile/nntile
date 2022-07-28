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
class Tile: public TileTraits, public StarpuNDimHandle
{
    std::vector<uint32_t> _starpu_shape()
    {
        // Convert data for StarPU
        std::vector<uint32_t> starpu_shape(ndim);
        for(size_t i = 0; i < ndim; ++i)
        {
            starpu_shape[i] = shape[i];
            // Check if conversion did not change values
            if(shape[i] != starpu_shape[i])
            {
                throw std::runtime_error("starpu_ndim_interface supports only "
                        "uint32_t type, which is not enough to hold provided "
                        "shape input");
            }
        }
        return starpu_shape;
    }
    std::vector<uint32_t> _starpu_stride()
    {
        // Convert data for StarPU
        std::vector<uint32_t> starpu_stride(ndim);
        for(size_t i = 0; i < ndim; ++i)
        {
            starpu_stride[i] = stride[i];
            // Check if conversion did not change values
            if(stride[i] != starpu_stride[i])
            {
                throw std::runtime_error("starpu_ndim_interface supports only "
                        "uint32_t type, which is not enough to hold provided "
                        "stride input");
            }
        }
        return starpu_stride;
    }
    uintptr_t _check_ptr(T *ptr, Index ptr_nelems)
    {
        if(nelems > ptr_nelems)
        {
            throw std::runtime_error("Required memory size is larger than "
                    "actually allocated memory");
        }
        return reinterpret_cast<uintptr_t>(ptr);
    }
public:
    Tile(const std::vector<Index> &shape_):
        TileTraits(shape_),
        StarpuNDimHandle(_starpu_shape(), _starpu_stride(), sizeof(T))
    {
    }
    Tile(const TileTraits &traits):
        TileTraits(traits),
        StarpuNDimHandle(_starpu_shape(), _starpu_stride(), sizeof(T))
    {
    }
    Tile(const std::vector<Index> &shape_, T *ptr, Index ptr_nelems):
        TileTraits(shape_),
        StarpuNDimHandle(_check_ptr(ptr, ptr_nelems), _starpu_shape(),
                _starpu_stride(), sizeof(T))
    {
    }
    Tile(const TileTraits &traits, T *ptr, Index ptr_nelems):
        TileTraits(traits),
        StarpuNDimHandle(_check_ptr(ptr, ptr_nelems), _starpu_shape(),
                _starpu_stride(), sizeof(T))
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

