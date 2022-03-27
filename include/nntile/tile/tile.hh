#pragma once

#include <nntile/base_type.hh>
#include <nntile/tile/traits.hh>
#include <starpu.h>

namespace nntile
{

struct TileHandle: public TileTraits
{
    //! Starpu handle for tile data
    StarPUSharedHandle handle;
    //! Underlying data type
    BaseType dtype;
    //! Constructor
    TileHandle(const std::vector<int> &shape_,
            const StarPUSharedHandle &handle_,
            const BaseType &dtype_):
        TileTraits(shape_),
        handle(handle_),
        dtype(dtype_)
    {
    }
    //! Constructor
    TileHandle(const TileTraits &traits,
            const StarPUSharedHandle &handle_,
            const BaseType &dtype_):
        TileTraits(traits),
        handle(handle_),
        dtype(dtype_)
    {
    }
};

template<typename T>
struct Tile: public TileHandle
{
    // No new member fields
    //! Constructor
    Tile(const std::vector<int> &shape_,
            const StarPUSharedHandle &handle_):
        TileHandle(shape_, handle_, BaseType(T{0}))
    {
    }
    //! Constructor
    Tile(const TileTraits &traits,
            const StarPUSharedHandle &handle_):
        TileHandle(traits, handle_, BaseType(T{0}))
    {
    }
};

} // namespace nntile

