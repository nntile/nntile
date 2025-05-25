/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/handle.hh
 * Wrappers for StarPU data handles
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <stdexcept>
#include <memory>

// Third-party headers
#include <starpu.h>

// Other NNTile headers

namespace nntile::starpu
{

//! Push a data handle into the list of registered data handles
void data_handle_push(starpu_data_handle_t handle);

//! Pop a data handle from the list of registered data handles
/* If the data handle is found in the list of registered data handles,
 * the function removes it from the list and returns the data handle.
 * Otherwise, the function returns a nullptr. */
starpu_data_handle_t data_handle_pop(starpu_data_handle_t handle);

//! Unregister all data handles
/* This function is called when the NNTile context is shut down. */
void data_handle_unregister_all();

// Forward declaration
class HandleLocalData;

//! StarPU data handle wrapper
class Handle: public std::shared_ptr<_starpu_data_state>
{
    //! Custom deleter for StarPU data handle uses async unregistration
    static void _deleter_async(starpu_data_handle_t handle);

public:
    //! Default constructor with nullptr
    Handle() = default;

    //! Constructor with shared pointer
    explicit Handle(starpu_data_handle_t handle):
        std::shared_ptr<_starpu_data_state>(handle, _deleter_async)
    {
        data_handle_push(handle);
    }

    //! Destructor uses async unregistration
    virtual ~Handle()
    {
        unregister_submit();
    }

    //! Set name of the handle
    void set_name(const char *name);

    //! Acquire data in CPU RAM
    HandleLocalData acquire(starpu_data_access_mode mode) const;

    //! Unregister a data handle normally
    void unregister();

    //! Unregister a data handle without coherency
    void unregister_no_coherency();

    //! Unregister a data handle in an async manner
    void unregister_submit();

    //! Invalidate data handle in an async manner
    void invalidate_submit() const;

    //! Get rank of the MPI node owning the data handle
    int mpi_get_rank() const;

    //! Get tag of the data handle
    int mpi_get_tag() const;

    //! Transfer data to a provided node rank
    void mpi_transfer(int dst_rank, int mpi_rank) const;

    //! Flush cached data
    void mpi_flush() const;
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
    void acquire(starpu_data_access_mode mode);

    //! Try to acquire the data in a non-blocking manner
    bool try_acquire(starpu_data_access_mode mode);

    //! Release the data
    void release();

    //! Get pointer to the local data
    void *get_ptr() const
    {
        return ptr;
    }
};

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
    static void _check_prereqs(size_t size);

    //! Register variable for StarPU-owned memory
    static starpu_data_handle_t _reg_data(size_t size);

    //! Register variable for user-owned memory
    static starpu_data_handle_t _reg_data(void *ptr, size_t size);
public:

    //! Constructor for variable that is managed by StarPU
    explicit VariableHandle(size_t size):
        Handle((_check_prereqs(size), _reg_data(size)))
    {
    }

    //! Constructor for variable that is managed by user
    explicit VariableHandle(void *ptr, size_t size):
        Handle((_check_prereqs(size), _reg_data(ptr, size)))
    {
    }
};

} // namespace nntile::starpu
