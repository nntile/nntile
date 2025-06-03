/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/prod_inplace.cc
 * Per-element product of two StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/prod_inplace.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/prod_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
ProdInplace<std::tuple<T>>::ProdInplace():
    codelet("nntile_prod_inplace", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply prod on StarPU buffers on CPU
template<typename T>
void ProdInplace<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::prod_inplace::cpu<T>(args->nelems, src, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply prod on StarPU buffer on CUDA
template<typename T>
void ProdInplace<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::prod_inplace::cuda<T>(stream, args->nelems, src, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for prod_inplace tasks
template<typename T>
uint32_t ProdInplace<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->nelems, sizeof(args->nelems), hash);
    return hash;
}

//! Submit prod_inplace task
template<typename T>
void ProdInplace<std::tuple<T>>::submit(Index nelems, Handle src, Handle dst)
{
    // Codelet arguments
    args_t *args = (args_t*)std::malloc(sizeof(*args));
    args->nelems = nelems;
    // Put amount of read-write bytes into flop count
    double nflops = sizeof(T) * 3 * nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_R, src.get(),
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in prod_inplace task submission");
    }
}

//! Pack of prod_inplace operations for different types
prod_inplace_pack_t prod_inplace;

} // namespace nntile::starpu
