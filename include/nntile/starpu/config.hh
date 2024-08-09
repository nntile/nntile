/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/starpu/config.hh
 * StarPU initialization/finalization and smart data handles
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <starpu.h>
// Disabled MPI for now
//#include <starpu_mpi.h>
#include <nntile/defs.h>
#include <nntile/logger/logger_thread.hh>

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

//! Convenient StarPU initialization and shutdown
class Config: public starpu_conf
{
    int cublas;
    int verbose;
public:
    explicit Config(int ncpus_=-1, int ncuda_=-1, int cublas_=-1, int logger=0,
            const char *logger_server_addr="localhost",
            int logger_server_port=5001, int verbose_=0)
    {
        starpu_fxt_autostart_profiling(0);
        // Init StarPU configuration with default values at first
        int ret = starpu_conf_init(this);
        if(ret != 0)
        {
            throw std::runtime_error("starpu_conf_init error");
        }
        // Set number of workers
        ncpus = ncpus_;
#ifdef NNTILE_USE_CUDA
        ncuda = ncuda_;
#else // NNTILE_USE_CUDA
        ncuda = 0;
#endif // NNTILE_USE_CUDA
        // Set history-based scheduler to utilize performance models
        sched_policy_name = "dmda";
        // Save initial value
        cublas = cublas_;
        // Verbosity level
        verbose = verbose_;
        // Init StarPU (master-slave)
        ret = starpu_init(this);
        if(ret != 0)
        {
            throw std::runtime_error("Error in starpu_initialize()");
        }
        else
        {
            int ncpus_ = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
            int ncuda_ = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
            if(verbose > 0)
            {
                std::cout << "Initialized NCPU=" << ncpus_ << " NCUDA=" <<
                    ncuda_ << "\n";
            }
        }
#ifdef NNTILE_USE_CUDA
        if(cublas != 0)
        {
            starpu_cublas_init();
            if(verbose > 0)
            {
                std::cout << "Initialized cuBLAS\n";
            }
        }
#endif // NNTILE_USE_CUDA
       if(logger != 0)
       {
           nntile::logger::logger_init(logger_server_addr, logger_server_port);
       }
    }
    ~Config()
    {
        shutdown();
    }
    void shutdown()
    {
        if(nntile::logger::logger_running)
        {
            nntile::logger::logger_shutdown();
        }
#ifdef NNTILE_USE_CUDA
        if(cublas != 0)
        {
            starpu_cublas_shutdown();
            if(verbose > 0)
            {
                std::cout << "Shutdown cuBLAS\n";
            }
        }
#endif // NNTILE_USE_CUDA
        starpu_shutdown();
        if(verbose > 0)
        {
            std::cout << "Shutdown StarPU\n";
        }
    }
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

// Forward declaration
class HandleLocalData;

//! StarPU data handle as a shared pointer to its internal state
//
// This class takes the ownership of the data handle. That said, it unregisters
// the data handle automatically at the end of lifetime.
class Handle
{
    // Different deleters for the handle
    static void _deleter(starpu_data_handle_t ptr)
    {
        // Unregister data and bring back result
        // All the tasks using given starpu data handle shall be finished
        // before unregistering the handle
        //std::cerr << "[nntile] unregister\n";
        starpu_data_unregister(ptr);
    }
    static void _deleter_no_coherency(starpu_data_handle_t ptr)
    {
        // Unregister data without bringing back result
        // All the tasks using given starpu data handle shall be finished
        // before unregistering the handle
        //std::cerr << "[nntile] unregister_no_coherency\n";
        starpu_data_unregister_no_coherency(ptr);
    }
    static void _deleter_temporary(starpu_data_handle_t ptr)
    {
        // Lazily unregister data as it is defined as temporary and may still
        // be in use. This shall only appear in use for data, allocated by
        // starpu as it will be deallocated during actual unregistering and at
        // the time of submission.
        //std::cerr << "[nntile] unregister_submit\n";
        starpu_data_unregister_submit(ptr);
    }
    static std::shared_ptr<_starpu_data_state> _get_shared_ptr(
            starpu_data_handle_t ptr, starpu_data_access_mode mode)
    {
        switch(mode)
        {
            case STARPU_R:
                //std::cerr << "[nntile] register READ\n";
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter_no_coherency);
            case STARPU_RW:
            case STARPU_W:
                //std::cerr << "[nntile] register WRITE\n";
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter);
            case STARPU_SCRATCH:
                //std::cerr << "[nntile] register SCRATCH\n";
                return std::shared_ptr<_starpu_data_state>(ptr,
                        _deleter_temporary);
            default:
                throw std::runtime_error("Invalid value of mode");
        }
    }
public:
    //! Shared handle itself
    std::shared_ptr<_starpu_data_state> handle;
    //! Default constructor with nullptr
    Handle():
        handle(nullptr)
    {
    }
    //! Constructor with shared pointer
    explicit Handle(std::shared_ptr<_starpu_data_state> handle_):
        handle(handle_)
    {
    }
    //! Constructor owns registered handle and unregisters it when needed
    explicit Handle(starpu_data_handle_t handle_,
            starpu_data_access_mode mode):
        handle(_get_shared_ptr(handle_, mode))
    {
    }
    //! Destructor is virtual as this is a base class
    virtual ~Handle()
    {
    }
    //! Convert to starpu_data_handle_t (only if explicitly asked)
    explicit operator starpu_data_handle_t() const
    {
        return handle.get();
    }
    //! Acquire data locally
    HandleLocalData acquire(starpu_data_access_mode mode) const;
    //! Unregister underlying handle without waiting for destructor
    void unregister()
    {
        handle.reset();
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
public:
    explicit HandleLocalData(const Handle &handle_,
            starpu_data_access_mode mode):
        handle(handle_)
    {
        acquire(mode);
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
        auto starpu_handle = static_cast<starpu_data_handle_t>(handle);
        int status = starpu_data_acquire(starpu_handle, mode);
        if(status != 0)
        {
            throw std::runtime_error("status != 0");
        }
        acquired = true;
        ptr = starpu_data_get_local_ptr(starpu_handle);
    }
    void release()
    {
        starpu_data_release(static_cast<starpu_data_handle_t>(handle));
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
    return HandleLocalData(*this, mode);
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
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, -1, 0, size);
        return tmp;
    }
    //! Register variable
    static starpu_data_handle_t _reg_data(void *ptr, size_t size)
    {
        if(size == 0)
        {
            throw std::runtime_error("Zero size is not supported");
        }
        starpu_data_handle_t tmp;
        starpu_variable_data_register(&tmp, STARPU_MAIN_RAM,
                reinterpret_cast<uintptr_t>(ptr), size);
        return tmp;
    }
public:
    //! Constructor for variable that is (de)allocated by StarPU
    explicit VariableHandle(size_t size, starpu_data_access_mode mode):
        Handle(_reg_data(size), mode)
    {
    }
    //! Constructor for variable that is (de)allocated by user
    explicit VariableHandle(void *ptr, size_t size,
            starpu_data_access_mode mode):
        Handle(_reg_data(ptr, size), mode)
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

} // namespace config
} // namespace nntile
