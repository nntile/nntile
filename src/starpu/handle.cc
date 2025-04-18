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

// Related headers
#include "nntile/starpu/handle.hh"

// Standard library headers
#include <mutex>
#include <unordered_set>

// Third-party headers
#include <starpu.h>
// Disabled MPI for now
//#include <starpu_mpi.h>

// Other NNTile headers

namespace nntile::starpu
{

//! Container for all registered data handles
std::unordered_set<starpu_data_handle_t> data_handles;

//! Mutex for a case of multi-threaded garbage collection
std::mutex data_handles_mutex;

//! Push a data handle into the list of registered data handles
void data_handle_push(starpu_data_handle_t handle)
{
    // Check if the StarPU is initialized
    if(starpu_is_initialized())
    {
        throw std::runtime_error("StarPU is not initialized");
    }

    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    // Put the data handle into the list of registered data handles
    data_handles.insert(handle);
}

//! Pop a data handle from the list of registered data handles
/* If the data handle is found in the list of registered data handles,
 * the function removes it from the list and returns the data handle.
 * Otherwise, the function returns a nullptr. */
starpu_data_handle_t data_handle_pop(starpu_data_handle_t handle)
{
    // Check if the StarPU is initialized
    if(starpu_is_initialized())
    {
        throw std::runtime_error("StarPU is not initialized");
    }

    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    // Find the data handle in the container
    auto it = data_handles.find(handle);
    if(it == data_handles.end())
    {
        // If the data handle is not found, return nullptr
        return nullptr;
    }

    // Erase the data handle from the container
    data_handles.erase(it);

    // Return the data handle
    return handle;
}

//! Unregister all data handles
/* This function is called when the NNTile context is shut down. */
void data_handle_unregister_all()
{
    // Check if the StarPU is initialized
    if(starpu_is_initialized())
    {
        throw std::runtime_error("StarPU is not initialized");
    }

    // Lock the data handles mutex to avoid race condition
    const std::lock_guard<std::mutex> lock(data_handles_mutex);

    // Unregister all data handles
    for(auto handle: data_handles)
    {
        starpu_data_unregister(handle);
    }

    // Clear the list of registered data handles
    data_handles.clear();
}

//! Deleter for the shared pointer uses async unregistration
void Handle::_deleter(starpu_data_handle_t handle)
{
    // Check if the data handle is still registered
    if(data_handle_pop(handle) != nullptr)
    {
        // Unregister the data handle in an async manner
        starpu_data_unregister_submit(handle);
    }
}

//! Set name of the handle
void Handle::set_name(const char *name)
{
    starpu_data_set_name(get(), name);
}


//! Acquire the data in a blocking manner
HandleLocalData Handle::acquire(starpu_data_access_mode mode) const
{
    return HandleLocalData(*this, mode, true);
}

//! Unregister a data handle normally
void Handle::unregister()
{
    // Only unregister if the data handle is still registered
    if(data_handle_pop(get()) != nullptr)
    {
        // Unregister the data handle in an async manner
        starpu_data_unregister_submit(get());
        // Reset the shared pointer
        reset();
    }
}

//! Unregister a data handle without coherency
void Handle::unregister_no_coherency()
{
    // Only unregister if the data handle is still registered
    if(data_handle_pop(get()) != nullptr)
    {
        // Unregister the data handle in an async manner
        starpu_data_unregister_submit(get());
        // Reset the shared pointer
        reset();
    }
}

//! Unregister a data handle in an async manner
void Handle::unregister_submit()
{
    // Only unregister if the data handle is still registered
    if(data_handle_pop(get()) != nullptr)
    {
        // Unregister the data handle in an async manner
        starpu_data_unregister_submit(get());
    }
}

//! Invalidate data handle in an async manner
void Handle::invalidate_submit() const
{
    starpu_data_invalidate_submit(get());
}

//! Get rank of the MPI node owning the data handle
int Handle::mpi_get_rank() const
{
    return 0;
    //return starpu_mpi_data_get_rank(get());
}

//! Get tag of the data handle
int Handle::mpi_get_tag() const
{
    return 0;
    //return starpu_mpi_data_get_tag(get());
}

//! Transfer data to a provided node rank
void Handle::mpi_transfer(int dst_rank, int mpi_rank) const
{
    //if(mpi_rank == dst_rank or mpi_rank == mpi_get_rank())
    //{
    //    // This function shall be removed in near future, all data
    //    // transfers shall be initiated by starpu_mpi_task_build and others
    //    starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
    //            get(), dst_rank, nullptr, nullptr);
    //    //int ret = starpu_mpi_get_data_on_node_detached(MPI_COMM_WORLD,
    //    //        get(), dst_rank, nullptr, nullptr);
    //    //if(ret != 0)
    //    //{
    //    //    throw std::runtime_error("Error in starpu_mpi_get_data_on_"
    //    //            "node_detached");
    //    //}
    //}
}

//! Flush cached data
void Handle::mpi_flush() const
{
    //starpu_mpi_cache_flush(MPI_COMM_WORLD, get());
}

//! Acquire the data in a blocking manner
void HandleLocalData::acquire(starpu_data_access_mode mode)
{
    // Acquire the data in a blocking manner
    auto starpu_handle = handle.get();
    int status = starpu_data_acquire(starpu_handle, mode);

    // Check if the data is acquired
    if(status != 0)
    {
        throw std::runtime_error("status != 0");
    }
    acquired = true;

    // Get pointer to the local data
    ptr = starpu_data_get_local_ptr(starpu_handle);
}

//! Try to acquire the data in a non-blocking manner
bool HandleLocalData::try_acquire(starpu_data_access_mode mode)
{
    // Early exit if the data is already acquired
    if (acquired) {
        return true;
    }

    // Acquire the data in a non-blocking manner
    auto starpu_handle = handle.get();
    int status = starpu_data_acquire_try(starpu_handle, mode);

    // Check if the data is acquired
    if(status != 0)
    {
        return false;
    }

    // Get pointer to the local data
    acquired = true;
    ptr = starpu_data_get_local_ptr(starpu_handle);
    return true;
}

//! Release the data
void HandleLocalData::release()
{
    starpu_data_release(handle.get());
    acquired = false;
    ptr = nullptr;
}

//! Check prereqs for registration
void VariableHandle::_check_prereqs(size_t size)
{
    // Check if StarPU is initialized
    if(!starpu_is_initialized())
    {
        throw std::runtime_error("StarPU is not initialized");
    }

    // Check if size is positive
    if(size == 0)
    {
        throw std::runtime_error("Zero size is not supported");
    }
}

//! Register variable for StarPU-owned memory
starpu_data_handle_t VariableHandle::_reg_data(size_t size)
{
    // Register the data handle
    starpu_data_handle_t tmp;
    starpu_variable_data_register(&tmp, -1, 0, size);
    return tmp;
}

//! Register variable for user-owned memory
starpu_data_handle_t VariableHandle::_reg_data(void *ptr, size_t size)
{
    // Register the data handle
    starpu_data_handle_t tmp;
    starpu_variable_data_register(&tmp, STARPU_MAIN_RAM,
            reinterpret_cast<uintptr_t>(ptr), size);
    return tmp;
}

} // namespace nntile::starpu
