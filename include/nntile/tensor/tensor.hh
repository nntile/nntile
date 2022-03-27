#pragma once

#include <nntile/tensor/traits.hh>
#include <memory>

namespace nntile
{

struct TensorHandle: public TensorTraits
{
    //! Pointer to the contiguous memory
    std::shared_ptr<std::byte> ptr;
    //! Data type
    BaseType dtype;
    //! Total size of allocated memory
    size_t alloc_size;
    //! Starpu handles for tile data
    std::vector<StarPUSharedHandle> tiles_handle;
    //! Constructor
    TensorHandle(const TensorTraits &traits,
            BaseType dtype_,
            int alignment=16):
        TensorTraits(traits),
        dtype(dtype_),
        alloc_size(0)
    {
        // At first compute memory footprint and offsets for each tile
        size_t elem_size = dtype.size();
        std::vector<size_t> tiles_offset(ntiles);
        for(int i = 0; i < ntiles; ++i)
        {
            // Store offset in bytes
            tiles_offset[i] = alloc_size;
            // Actual memory for the tile in bytes
            size_t tile_alloc_size = elem_size * tiles_traits[i].nelems;
            // Round up to the alignment parameter
            size_t naligns = (tile_alloc_size-1)/alignment + 1;
            alloc_size += naligns * alignment;
        }
        //std::cout << "tile_traits.size()=" << tiles_traits.size() << "\n";
        // Allocate memory
        //std::cout << "alloc_size=" << alloc_size << "\n";
        auto ptr_raw = ::new std::byte[alloc_size];
        ptr = std::shared_ptr<std::byte>(ptr_raw);
        uintptr_t ptr_uint = reinterpret_cast<uintptr_t>(ptr_raw);
        // Register tiles
        tiles_handle.reserve(ntiles);
        for(int i = 0; i < ntiles; ++i)
        {
            tiles_handle.emplace_back(STARPU_MAIN_RAM,
                    ptr_uint+tiles_offset[i], tiles_traits[i].nelems,
                    elem_size);
        }
    }
    //! Constructor
    TensorHandle(const std::vector<int> &shape_,
            const std::vector<int> &tile_shape_,
            BaseType dtype_,
            int alignment=16):
        TensorHandle({shape_, tile_shape_}, dtype_, alignment)
    {
    }
};

template<typename T>
struct Tensor: public TensorHandle
{
    // No new member fields
    //! Constructor
    Tensor(const TensorTraits &traits):
        TensorHandle(traits, BaseType(T{0}))
    {
    }
    //! Constructor
    Tensor(const std::vector<int> &shape_,
            const std::vector<int> &tile_shape_):
        TensorHandle(shape_, tile_shape_, BaseType(T{0}))
    {
    }
};

} // namespace nntile

