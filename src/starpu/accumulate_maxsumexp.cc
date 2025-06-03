/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate_maxsumexp.cc
 * Accumulate one StarPU maxsumexp buffer into another
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/accumulate_maxsumexp.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/accumulate_maxsumexp.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AccumulateMaxSumExp<std::tuple<T>>::AccumulateMaxSumExp():
    codelet("nntile_accumulate_maxsumexp", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply accumulate_maxsumexp operation for StarPU buffers in CPU
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T) / 2;
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::accumulate_maxsumexp::cpu<T>(nelems, src, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply accumulate_maxsumexp for StarPU buffers on CUDA
template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T) / 2;
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::accumulate_maxsumexp::cuda<T>(stream, nelems, src, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AccumulateMaxSumExp<std::tuple<T>>::submit(Handle src, Handle dst)
//! Insert accumulate_maxsumexp task into StarPU pool of tasks
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
        throw std::runtime_error("Error in accumulate_maxsumexp task submission");
    }
}

//! Pack of accumulate_maxsumexp operations for different types
accumulate_maxsumexp_pack_t accumulate_maxsumexp;

} // namespace nntile::starpu
