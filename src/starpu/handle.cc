/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/handle.cc
 * Wrappers for StarPU data handles
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>
#include <starpu.h>
// Disabled MPI for now
//#include <starpu_mpi.h>
#include <nntile/defs.h>
#include <nntile/context.hh>

namespace nntile::starpu
{

// Forward declaration
class HandleLocalData;

//! StarPU data handle wrapper
class Handle
{
public:
    //! Shared handle itself
    starpu_data_handle_t handle = nullptr;

    //! NNTile context
    Context &context;

    //! Default constructor with nullptr
    Handle() = default;

    //! Constructor with shared pointer
    explicit Handle(starpu_data_handle_t handle_, Context &context_):
        handle(handle_), context(context_)
    {
        context.data_handle_register(handle);
    }

    //! Destructor is virtual as this is a base class
    virtual ~Handle()
    {
    }

    //! Get the handle itself
    starpu_data_handle_t get() const
    {
        return handle;
    }

    //! Set name of the handle
    void set_name(const char *name)
    {
        starpu_data_set_name(handle, name);
    }

    //! Acquire data locally
    HandleLocalData acquire(starpu_data_access_mode mode) const;

    //! Unregister a data handle normally
    void unregister()
    {
        // Only unregister if handle is not nullptr
        if(handle != nullptr)
        {
            context.data_handle_unregister(handle);
            handle = nullptr;
        }
    }

    //! Unregister a data handle without coherency
    void unregister_no_coherency()
    {
        // Only unregister if handle is not nullptr
        if(handle != nullptr)
        {
            context.data_handle_unregister_no_coherency(handle);
            handle = nullptr;
        }
    }

    //! Unregister a data handle in an async manner
    void unregister_submit()
    {
        // Only unregister if handle is not nullptr
        if(handle != nullptr)
        {
            context.data_handle_unregister_submit(handle);
            handle = nullptr;
        }
    }

    //! Invalidate data handle in an async manner
    void invalidate_submit() const
    {
        starpu_data_invalidate_submit(handle);
    }

    //! Get rank of the MPI node owning the data handle
    int mpi_get_rank() const
    {
        return 0;
        //return starpu_mpi_data_get_rank(handle.get());
    }

    //! Get tag of the data handle
    int mpi_get_tag() const
    {
        return 0;
        //return starpu_mpi_data_get_tag(handle.get());
    }

    //! Transfer data to a provided node rank
    void mpi_transfer(int dst_rank, int mpi_rank) const
    {
        //if(mpi_rank == dst_rank or mpi_rank == mpi_get_rank())
        //{
        //    // This function shall be removed in near future, all data
        //    // transfers shall be initiated by starpu_mpi_task_build and others
        //    starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
        //            handle.get(), dst_rank, nullptr, nullptr);
        //    //int ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
        //    //        handle.get(), dst_rank, nullptr, nullptr);
        //    //if(ret != 0)
        //    //{
        //    //    throw std::runtime_error("Error in starpu_mpi_get_data_on_"
        //    //            "node_detached");
        //    //}
        //}
    }

    //! Flush cached data
    void mpi_flush() const
    {
        //starpu_mpi_cache_flush(MPI_COMM_WORLD, handle.get());
    }
};

//! Local data handle wrapper
class HandleLocalData
{
public:
    //! Handle
    Handle handle;

    //! Pointer to the local data
    void *ptr = nullptr;

    //! Whether the data is acquired
    bool acquired = false;

    //! Whether the data is blocking
    bool is_blocking_ = true;

    //! Constructor
    explicit HandleLocalData(const Handle &handle_,
            starpu_data_access_mode mode, bool is_blocking = true):
        handle(handle_), is_blocking_(is_blocking)
    {
        if(is_blocking)
        {
            acquire(mode);
        }
    }

    //! Destructor is virtual as this is a derived class
    virtual ~HandleLocalData()
    {
        if(acquired)
        {
            release();
        }
    }

    //! Acquire the data in a blocking manner
    void acquire(starpu_data_access_mode mode)
    {
        auto starpu_handle = handle.get();
        int status = starpu_data_acquire(starpu_handle, mode);
        if(status != 0)
        {
            throw std::runtime_error("status != 0");
        }
        acquired = true;
        ptr = starpu_data_get_local_ptr(starpu_handle);
    }

    //! Try to acquire the data in a non-blocking manner
    bool try_acquire(starpu_data_access_mode mode)
    {
        if (acquired) {
            return true;
        }

        auto starpu_handle = handle.get();
        int status = starpu_data_acquire_try(starpu_handle, mode);
        if(status != 0)
        {
            return false;
        }

        acquired = true;
        ptr = starpu_data_get_local_ptr(starpu_handle);
        return true;
    }

    //! Release the data
    void release()
    {
        starpu_data_release(handle.get());
        acquired = false;
        ptr = nullptr;
    }

    //! Get pointer to the local data
    void *get_ptr() const
    {
        return ptr;
    }
};

//! Acquire the data in a blocking manner
inline
HandleLocalData Handle::acquire(starpu_data_access_mode mode) const
{
    return HandleLocalData(*this, mode, true);
}

//! Wrapper for struct starpu_variable_interface
class VariableInterface: public starpu_variable_interface
{
public:
    //! No constructor
    VariableInterface() = delete;

    //! No destructor
    ~VariableInterface() = delete;

    //! Get pointer of a proper type
    template<typename T>
    T *get_ptr() const
    {
        return reinterpret_cast<T *>(ptr);
    }
};

//! Convenient registration and deregistration of data through StarPU handle
class VariableHandle: public Handle
{
    //! Check prereqs for registration
    static void _check_prereqs(size_t size, const Context &context)
    {
        // Check if context is initialized
        if(context.initialized == 0)
        {
            throw std::runtime_error("Context is not initialized, cannot register"
                " data handle");
        }
        // Check if size is positive
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
    }
    //! Register variable for StarPU-owned memory
    static starpu_data_handle_t _reg_data(size_t size)
    {
        // Register the data handle
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, -1, 0, size);
        return tmp;
    }

    //! Register variable for user-owned memory
    static starpu_data_handle_t _reg_data(void *ptr, size_t size,
            const Context &context)
    {
        // Check if context is initialized
        if(context.initialized == 0)
        {
            throw std::runtime_error("Context is not initialized, cannot register"
                " data handle");
        }
        // Check if size is positive
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        // Register the data handle
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, STARPU_MAIN_RAM,
                reinterpret_cast<uintptr_t>(ptr), size);
        return tmp;
    }
public:
    //! Constructor for variable that is managed by StarPU
    explicit VariableHandle(size_t size, Context &context):
        Handle((_check_prereqs(size, context), _reg_data(size)), context)
    {
    }

    //! Constructor for variable that is managed by user
    explicit VariableHandle(void *ptr, size_t size, Context &context):
        Handle((_check_prereqs(size, context), _reg_data(ptr, size)), context)
    {
    }
};

} // namespace nntile::starpu
