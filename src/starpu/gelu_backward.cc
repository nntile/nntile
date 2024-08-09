/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu_backward.cc
 * Backward GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/gelu_backward.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/gelu_backward.hh"
#include <cstdlib>

namespace nntile::starpu::gelu_backward
{

//! Apply backward GeLU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *x = interfaces[0]->get_ptr<T>();
    const T *dy = interfaces[1]->get_ptr<T>();
    T *dx = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::gelu_backward::cpu<T>(nelems, x, dy, dx);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply GeLU backward on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *x = interfaces[0]->get_ptr<T>();
    const T *dy = interfaces[1]->get_ptr<T>();
    T *dx = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::gelu_backward::cuda<T>(stream, nelems, x, dy, dx);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_gelu_backward_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_gelu_backward_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index nelems, Handle x, Handle dy, Handle dx)
{
    Index *nelems_ = (Index *)std::malloc(sizeof(*nelems_));
    *nelems_ = nelems;
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(x),
            STARPU_R, static_cast<starpu_data_handle_t>(dy),
            STARPU_RW, static_cast<starpu_data_handle_t>(dx),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelu_backward task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Index nelems, Handle x, Handle dy, Handle dx);

template
void submit<fp64_t>(Index nelems, Handle x, Handle dy, Handle dx);

template<typename T>
void submit_mpi(Index nelems, Handle x, Handle dy, Handle dx, int exec_rank)
{
    // Build a task with initializing data transfers
    struct starpu_task *task = starpu_task_build(
            codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(x),
            STARPU_R, static_cast<starpu_data_handle_t>(dy),
            STARPU_RW, static_cast<starpu_data_handle_t>(dx),
            0);
    // Only execution node will have non-nullptr task
    if(task)
    {
        // Define codelet arguments
        Index *nelems_ = (Index *)std::malloc(sizeof(*nelems_));
        task->cl_arg = nelems_;
        task->cl_arg_size = sizeof(*nelems_);
        task->cl_arg_free = 1;
        // Set codelet arguments
        *nelems_ = nelems;
        // Submit task to the DAG
        int ret = starpu_task_submit(task);
        // Check submission
        if(ret != 0)
        {
            throw std::runtime_error("Error in gelu_backward MPI task "
                    "submission");
        }
    }
    // Data transfers after the task
//    starpu_mpi_task_post_build(MPI_COMM_WORLD,
//            codelet<T>(),
//            STARPU_R, static_cast<starpu_data_handle_t>(x),
//            STARPU_R, static_cast<starpu_data_handle_t>(dy),
//            STARPU_RW, static_cast<starpu_data_handle_t>(dx),
//            STARPU_EXECUTE_ON_NODE, exec_rank,
//            0);

}

// Explicit instantiaion
template
void submit_mpi<fp32_t>(Index nelems, Handle x, Handle dy, Handle dx,
        int exec_rank);

template
void submit_mpi<fp64_t>(Index nelems, Handle x, Handle dy, Handle dx,
        int exec_rank);

} // namespace nntile::starpu::gelu_backward
