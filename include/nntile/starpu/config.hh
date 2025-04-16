/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/config.hh
 * StarPU configuration, data handles and codelets base classes
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <mutex>
#include <starpu.h>
// Disabled MPI for now
//#include <starpu_mpi.h>
#include <nntile/defs.h>

namespace nntile
{

// Fake STARPU functions
#define MPI_COMM_WORLD 0

static int starpu_mpi_world_size()
{
    return 1;
}

static int starpu_mpi_world_rank()
{
    return 0;
}

static int starpu_mpi_wait_for_all(int comm)
{
    return 0;
}

static int starpu_mpi_barrier(int comm)
{
    return 0;
}

namespace starpu
{

//! Convenient StarPU configuration class
class Config
{
    //! StarPU configuration without explicit default value
    // It will be zero-initialized due to global configuration variable
    starpu_conf starpu_config;
    //! Flag if StarPU codelets are initialized
    bool codelets_initialized = false;
    //! Whether cuBLAS is enabled with the configuration
    bool cublas = false;
    //! Whether Out-of-Core is enabled with the configuration
    bool ooc = false;
    //! Out-of-core path
    const char *ooc_path = nullptr;
    //! Out-of-core size
    size_t ooc_size = 16777216;
    //! Out-of-core disk node id
    int ooc_disk_node_id = -1;
    //! Verbosity level
    int verbose = 0;
    //! Container for all data handles
    std::unordered_set<starpu_data_handle_t> data_handles;
    //! Automatically unregistered data handles
    std::unordered_set<starpu_data_handle_t> auto_unreg_data_handles;
    //! Mutex for data handles
    std::mutex data_handles_mutex;
public:
    // Constructor is the default one
    Config() = default;

    //! Proper destructor for the only available configuration object
    ~Config()
    {
        // It is safe to call shutdown multiple times
        shutdown();
    }

    //! Initialize StarPU and NNTile with the configuration
    void init(
        int ncpus=-1,
        int ncuda=-1,
        int cublas_=1,
        int ooc_=0,
        const char *ooc_path_="/tmp/nntile_ooc",
        size_t ooc_size_=16777216,
        int ooc_disk_node_id_=-1,
        int verbose_=0
    );

    //! Shutdown StarPU
    void shutdown();

    //! Initialize StarPU codelets
    void init_codelets();

    //! Insert a data handle into the container
    void data_handle_register(starpu_data_handle_t handle);

    //! Pop a data handle from the container
    /* Tries to pop a data handle from the container of all registered data
     * handles. If the data is still registered, pops it from the container
     * and returns True. If the data handle is not registered, checks if it
     * was automatically unregistered and pops it from the auto-unregistered
     * container and returns False. Otherwise, throws an error. Return value
     * indicates if the data handle is still registered.
     */
    bool data_handle_pop(starpu_data_handle_t handle);

    //! Unregister a data handle from the container
    void data_handle_unregister(starpu_data_handle_t handle);

    //! Unregister a data handle without coherency
    void data_handle_unregister_no_coherency(starpu_data_handle_t handle);

    //! Unregister a data handle in an async manner
    void data_handle_unregister_submit(starpu_data_handle_t handle);

    //! StarPU commute data access mode
    static constexpr starpu_data_access_mode STARPU_RW_COMMUTE

    //    = STARPU_RW; // Temporarily disabled commute mode
        = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);

    // Unpack args by pointers without copying actual data
    template<typename... Ts>
    static
    void unpack_args_ptr(void *cl_args, const Ts *&...args)
    {
        // The first element is a total number of packed arguments
        int nargs = reinterpret_cast<int *>(cl_args)[0];
        cl_args = reinterpret_cast<char *>(cl_args) + sizeof(int);
        // Unpack arguments one by one
        if(nargs > 0)
        {
            unpack_args_ptr_single_arg(cl_args, nargs, args...);
        }
    }

    // Unpack with no argument remaining
    static
    void unpack_args_ptr_single_arg(void *cl_args, int nargs)
    {
    }

    // Unpack arguments one by one
    template<typename T, typename... Ts>
    static
    void unpack_args_ptr_single_arg(void *cl_args, int nargs, const T *&ptr,
            const Ts *&...args)
    {
        // Do nothing if there are no remaining arguments
        if(nargs == 0)
        {
            return;
        }
        // The first element is a size of argument
        size_t arg_size = reinterpret_cast<size_t *>(cl_args)[0];
        // Get pointer to the data
        char *char_ptr = reinterpret_cast<char *>(cl_args) + sizeof(size_t);
        ptr = reinterpret_cast<T *>(char_ptr);
        // Move pointer by data size
        cl_args = char_ptr + arg_size;
        // Unpack next argument
        unpack_args_ptr_single_arg(cl_args, nargs-1, args...);
    }
};

//! Global StarPU configuration object
extern Config config;

// Forward declaration
class HandleLocalData;

//! StarPU data handle wrapper
class Handle
{
public:
    //! Shared handle itself
    starpu_data_handle_t handle = nullptr;
    //! Default constructor with nullptr
    Handle() = default;
    //! Constructor with shared pointer
    explicit Handle(starpu_data_handle_t handle_):
        handle(handle_)
    {
        nntile::starpu::config.data_handle_register(handle);
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
            nntile::starpu::config.data_handle_unregister(handle);
            handle = nullptr;
        }
    }

    //! Unregister a data handle without coherency
    void unregister_no_coherency()
    {
        // Only unregister if handle is not nullptr
        if(handle != nullptr)
        {
            nntile::starpu::config.data_handle_unregister_no_coherency(handle);
            handle = nullptr;
        }
    }

    //! Unregister a data handle in an async manner
    void unregister_submit()
    {
        // Only unregister if handle is not nullptr
        if(handle != nullptr)
        {
            nntile::starpu::config.data_handle_unregister_submit(handle);
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

class HandleLocalData
{
    Handle handle;
    void *ptr = nullptr;
    bool acquired = false;
    bool is_blocking_ = true;
public:
    explicit HandleLocalData(const Handle &handle_,
            starpu_data_access_mode mode, bool is_blocking = true):
        handle(handle_), is_blocking_(is_blocking)
    {
        if (is_blocking_) {
            acquire(mode);
        }
    }
    virtual ~HandleLocalData()
    {
        if(acquired)
        {
            release();
        }
    }
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

    void release()
    {
        starpu_data_release(handle.get());
        acquired = false;
        ptr = nullptr;
    }
    void *get_ptr() const
    {
        return ptr;
    }
};

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
    //! Register variable for StarPU-owned memory
    static starpu_data_handle_t _reg_data(size_t size)
    {
        // Check if StarPU is initialized
        if(!starpu_is_initialized())
        {
            throw std::runtime_error("StarPU is not initialized, cannot register"
                " data handle");
        }
        // Check if size is positive
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        // Register the data handle
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, -1, 0, size);
        // Disable OOC by default
        starpu_data_set_ooc_flag(tmp, 0);
        return tmp;
    }
    //! Register variable
    static starpu_data_handle_t _reg_data(void *ptr, size_t size)
    {
        // Check if StarPU is initialized
        if(!starpu_is_initialized())
        {
            throw std::runtime_error("StarPU is not initialized, cannot register"
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
        // Disable OOC by default
        starpu_data_set_ooc_flag(tmp, 0);
        return tmp;
    }
public:
    //! Constructor for variable that is (de)allocated by StarPU
    explicit VariableHandle(size_t size):
        Handle(_reg_data(size))
    {
    }
    //! Constructor for variable that is (de)allocated by user
    explicit VariableHandle(void *ptr, size_t size):
        Handle(_reg_data(ptr, size))
    {
    }
};

//! StarPU codelet+perfmodel wrapper
class Codelet: public starpu_codelet, public starpu_perfmodel
{
private:
    uint32_t where_default = STARPU_NOWHERE; // uninitialized value
public:
    //! Zero-initialize codelet
    Codelet()
    {
        std::memset(this, 0, sizeof(*this));
    }
    void init(const char *name_, uint32_t (*footprint_)(starpu_task *),
            std::initializer_list<starpu_cpu_func_t> cpu_funcs_,
            std::initializer_list<starpu_cuda_func_t> cuda_funcs_)
    {
        // Initialize perfmodel
        starpu_codelet::model = this;
        starpu_perfmodel::type = STARPU_HISTORY_BASED;
        // Set codelet name and performance model symbol
        starpu_codelet::name = name_;
        starpu_perfmodel::symbol = name_;
        // Set footprint function
        starpu_perfmodel::footprint = footprint_;
        // Runtime decision on number of buffers and modes
        starpu_codelet::nbuffers = STARPU_VARIABLE_NBUFFERS;
        // Add CPU implementations
        if(cpu_funcs_.size() > STARPU_MAXIMPLEMENTATIONS)
        {
            throw std::runtime_error("Too many CPU func implementations");
        }
        if(cpu_funcs_.size() > 0)
        {
            auto it = cpu_funcs_.begin();
            for(int i = 0; i < cpu_funcs_.size(); ++i, ++it)
            {
                if(*it)
                {
//#ifdef STARPU_SIMGRID // Put fake function address in case of simulation
//                    starpu_codelet::cpu_funcs[i] = (starpu_cpu_func_t)0;
//#else // Put real function address
                    starpu_codelet::cpu_funcs[i] = *it;
//#endif
                    starpu_codelet::where = where_default = STARPU_CPU;
                }
            }
        }
        // Add CUDA implementations
        if(cuda_funcs_.size() > STARPU_MAXIMPLEMENTATIONS)
        {
            throw std::runtime_error("Too many CUDA func implementations");
        }
        if(cuda_funcs_.size() > 0)
        {
            auto it = cuda_funcs_.begin();
            for(int i = 0; i < cuda_funcs_.size(); ++i, ++it)
            {
                if(*it)
                {
//#ifdef STARPU_SIMGRID // Put fake function address in case of simulation
//                    starpu_codelet::cuda_funcs[i] = (starpu_cuda_func_t)0;
//#else // Put real function address
                    starpu_codelet::cuda_funcs[i] = *it;
//#endif
                    starpu_codelet::cuda_flags[i] = STARPU_CUDA_ASYNC;
                    where_default = where_default | STARPU_CUDA;
                    starpu_codelet::where = where_default;
                }
            }
        }
    }
    void restrict_where(uint32_t where_)
    {
        if((where_default & where_) == where_)
        {
            starpu_codelet::where = where_;
        }
        else
        {
            //throw std::runtime_error("Provided where is not supported");
        }
    }
    void restore_where()
    {
        starpu_codelet::where = where_default;
    }
};

} // namespace starpu
} // namespace nntile
