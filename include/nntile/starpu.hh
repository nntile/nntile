#pragma once

#include <starpu.h>
#include <stdexcept>
#include <memory>

namespace nntile
{

class StarPU
{
    struct starpu_conf conf;
public:
    StarPU()
    {
        int ret = starpu_init(nullptr);
        if(ret == -ENODEV)
        {
            throw std::runtime_error("Error in starpu_init");
        }
    }
    ~StarPU()
    {
        starpu_task_wait_for_all();
        starpu_shutdown();
    }
    StarPU(const StarPU &) = delete;
    StarPU(StarPU &&) = delete;
    StarPU &operator=(const StarPU &) = delete;
    StarPU &operator=(StarPU &&) = delete;
};

class StarPUHandle
{
    starpu_data_handle_t handle;
public:
    //! Constructor
    explicit StarPUHandle(int home_node,
            uintptr_t ptr,
            uint32_t nelems,
            size_t elem_size)
    {
        starpu_vector_data_register(&handle, home_node, ptr, nelems,
                elem_size);
    }
    //! Destructor
    ~StarPUHandle()
    {
        starpu_data_unregister(handle);
    }
    //! No copy constructor
    StarPUHandle(const StarPUHandle &) = delete;
    //! No move constructor
    StarPUHandle(StarPUHandle &&) = delete;
    //! No copy assignment
    StarPUHandle &operator=(const StarPUHandle &) = delete;
    //! No move assignment
    StarPUHandle &operator=(StarPUHandle &&) = delete;
    //! Convert to starpu_data_handle_t
    operator starpu_data_handle_t() const
    {
        return handle;
    }
};

class StarPUSharedHandle
{
    std::shared_ptr<StarPUHandle> shared_handle;
public:
    StarPUSharedHandle() = default;
    StarPUSharedHandle(int home_node,
            uintptr_t ptr,
            uint32_t nelems,
            size_t elem_size):
        shared_handle(::new StarPUHandle(home_node, ptr, nelems, elem_size))
    {
    }
    StarPUSharedHandle(int home_node,
            float *ptr,
            uint32_t nelems):
        StarPUSharedHandle(home_node, reinterpret_cast<uintptr_t>(ptr),
                nelems, sizeof(*ptr))
    {
    }
    StarPUSharedHandle(int home_node,
            float *ptr,
            int nelems):
        StarPUSharedHandle(home_node, reinterpret_cast<uintptr_t>(ptr),
                static_cast<uint32_t>(nelems), sizeof(*ptr))
    {
        if(nelems < 0)
        {
            throw std::runtime_error("nelems < 0");
        }
    }
    operator starpu_data_handle_t() const
    {
        if(!shared_handle)
        {
            throw std::runtime_error("shared_handle is nullptr");
        }
        return starpu_data_handle_t{shared_handle.get()[0]};
    };
};

} // namespace nntile

