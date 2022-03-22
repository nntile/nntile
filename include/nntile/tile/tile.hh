#pragma once

#include <nntile/base_type.hh>
#include <nntile/tile/traits.hh>
#include <starpu.h>

namespace nntile
{

struct BaseContiguousTile: public ContiguousTileTraits
{
    //! Raw pointer to the tile data
    void *ptr;
    //! Data type
    BaseType dtype;
    //! Size of a single tile element
    size_t elem_size;
    //! Total size of allocated memory
    size_t alloc_size;
    //! Starpu handle for tile data
    starpu_data_handle_t handle;
    //! Flag if the data is pinned
    int pinned;
    //! Constructor
    template<typename T_shape>
    BaseContiguousTile(const T_shape &shape_, void *ptr_,
            BaseType dtype_):
        ContiguousTileTraits(shape_),
        ptr(ptr_),
        dtype(dtype_),
        elem_size(dtype.size()),
        alloc_size(nelems * elem_size),
        handle(nullptr),
        pinned(0)
    {
    }
    //! Register starpu handle
    void data_register()
    {
        starpu_vector_data_register(&handle, STARPU_MAIN_RAM,
                reinterpret_cast<uintptr_t>(ptr), nelems,
                elem_size);
    }
    //! Unregister starpu handle
    void data_unregister()
    {
        starpu_data_unregister(handle);
        handle = nullptr;
    }
    //! Pin memory
    void data_pin()
    {
        starpu_memory_pin(ptr, alloc_size);
    }
    //! Unpin memory
    void data_unpin()
    {
        starpu_memory_unpin(ptr, alloc_size);
    }
    //! Offset
    size_t offset(const std::vector<int> &index) const
    {
        return elem_size * ContiguousTileTraits::offset(index);
    }
    template<size_t NDIM>
    size_t offset(const std::array<int, NDIM> &index) const
    {
        return elem_size * ContiguousTileTraits::offset(index);
    }
};

template<typename T>
struct ContiguousTile: public BaseContiguousTile
{
    //! Type of values
    using value_type = T;
    //! Constructor
    template<typename T_shape>
    ContiguousTile(const T_shape &shape_, T *ptr_):
        BaseContiguousTile(shape_, ptr_, ptr_)
    {
    }
    //! Pointer to the corresponding index
    template<typename T_index>
    T at(const T_index &index) const
    {
        return reinterpret_cast<T *>(ptr)[ContiguousTileTraits::offset(index)];
    }
};

} // namespace nntile

