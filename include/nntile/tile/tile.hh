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
    Tile(const std::vector<size_t> &shape_):
        TileTraits(shape_),
        StarpuVariableHandle(nelems*sizeof(T))
    {
    }
    Tile(const TileTraits &traits):
        TileTraits(traits),
        StarpuVariableHandle(nelems*sizeof(T))
    {
    }
    Tile(const std::vector<size_t> &shape_, T *ptr, size_t ptr_nelems):
        TileTraits(shape_),
        StarpuVariableHandle(reinterpret_cast<uintptr_t>(ptr),
                sizeof(T)*check_nelems(nelems, ptr_nelems))
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
};

} // namespace nntile

