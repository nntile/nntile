/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate_hypot.cc
 * Accumulate one StarPU buffers into another as hypot
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/accumulate_hypot.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AccumulateHypot<std::tuple<T>>::AccumulateHypot():
    codelet("nntile_accumulate_hypot", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply accumulate_hypot operation for StarPU buffers in CPU
template<typename T>
void AccumulateHypot<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T);
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::hypot::cpu<T>(nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply accumulate_hypot for StarPU buffers on CUDA
template<typename T>
void AccumulateHypot<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T);
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::hypot::cuda<T>(stream, nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AccumulateHypot<std::tuple<T>>::submit(Handle src, Handle dst)
//! Insert accumulate hypoy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW | STARPU_COMMUTE, dst.get(),
            STARPU_R, src.get(),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in accumulate_hypot task submission");
    }
}

//! Pack of accumulate_hypot operations for different types
accumulate_hypot_pack_t accumulate_hypot;

} // namespace nntile::starpu
