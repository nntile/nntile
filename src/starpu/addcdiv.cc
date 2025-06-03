/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/addcdiv.cc
 * Per-element addcdiv operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/addcdiv.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/addcdiv.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AddCdiv<std::tuple<T>>::AddCdiv():
    codelet("nntile_addcdiv", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply addcdiv operation on StarPU buffers on CPU
template<typename T>
void AddCdiv<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *nom = interfaces[0]->get_ptr<T>();
    const T *denom = interfaces[1]->get_ptr<T>();
    T *src = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::addcdiv::cpu<T>(
        args->val, args->eps, args->nelems, nom, denom, src);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply addcdiv on StarPU buffer on CUDA
template<typename T>
void AddCdiv<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *nom = interfaces[0]->get_ptr<T>();
    const T *denom = interfaces[1]->get_ptr<T>();
    T *src = interfaces[2]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::addcdiv::cuda<T>(
        stream, args->val, args->eps, args->nelems, nom, denom, src);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AddCdiv<std::tuple<T>>::submit(
    Index nelems,
    Scalar val,
    Scalar eps,
    Handle nom,
    Handle denom,
    Handle src
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->val = val;
    args->eps = eps;
    args->nelems = nelems;
    //double nflops = 5 * nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, nom.get(),
            STARPU_R, denom.get(),
            STARPU_RW, src.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in addcdiv task submission");
    }
}

//! Pack of addcdiv operations for different types
addcdiv_pack_t addcdiv;

} // namespace nntile::starpu
