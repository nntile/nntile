#pragma once

#include <nntile/tile/traits.hh>
#include <nntile/starpu.hh>

namespace nntile
{

template<typename T>
class Tile: public TileTraits, public StarpuVariableHandle
{
    size_t check_nelems(size_t req_nelems, size_t ptr_nelems)
    {
        if(req_nelems > ptr_nelems)
        {
            throw std::runtime_error("Required memory size is larger than "
                    "actually allocated memory");
        }
        return req_nelems;
    }
public:
    Tile(const TileTraits &traits):
        TileTraits(traits),
        StarpuVariableHandle(nelems*sizeof(T))
    {
    }
    Tile(const TileTraits &traits, T *ptr, size_t ptr_nelems):
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
    //! Read an element of the tile
    T at_index(const std::vector<size_t> &index) const
    {
        // Get the corresponding tile data to the local buffer
        acquire(STARPU_R);
        // Read the value, bounds are checked in index_to_linear() function
        T value = get_local_ptr()[index_to_linear(index)];
        // Release the buffer
        release();
        return value;
    }
    //! Read an element of the tile
    T at_linear(size_t linear_offset) const
    {
        // Check bounds
        if(linear_offset >= nelems)
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

