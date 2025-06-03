/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/hypot_scalar_inverse.cc
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/hypot_scalar_inverse.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_scalar_inverse.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
HypotScalarInverse<std::tuple<T>>::HypotScalarInverse():
    codelet("nntile_hypot_scalar_inverse", footprint, cpu_funcs, cuda_funcs)
{
}

//! Apply hypot_scalar_inverse operation for StarPU buffers in CPU
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *dst = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::hypot_scalar_inverse::cpu<T>(args->nelems, args->eps, args->alpha,
            dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply hypot_scalar_inverse for StarPU buffers on CUDA
template<typename T>
void HypotScalarInverse<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *dst = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::hypot_scalar_inverse::cuda<T>(stream, args->nelems, args->eps,
            args->alpha, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Submit hypot_scalar_inverse task
template<typename T>
void HypotScalarInverse<std::tuple<T>>::submit(
        Index nelems, Scalar eps, Scalar alpha, Handle dst)
//! Insert hypot_scalar_inverse task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->nelems = nelems;
    args->eps = eps;
    args->alpha = alpha;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, dst.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in hypot_scalar_inverse task submission");
    }
}

//! Pack of hypot_scalar_inverse operations for different types
hypot_scalar_inverse_pack_t hypot_scalar_inverse;

} // namespace nntile::starpu
