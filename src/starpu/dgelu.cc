/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/dgelu.cc
 * Derivative of GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/dgelu.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// NNTile headers
#include "nntile/kernel/dgelu.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
DGelu<std::tuple<T>>::DGelu():
    codelet("nntile_dgelu", footprint, cpu_funcs, cuda_funcs)
{
    codelet.set_modes_fixed({STARPU_RW});
}

//! Apply dgelu on StarPU buffer on CPU
template<typename T>
void DGelu<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::dgelu::cpu<T>(nelems, data);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply dgelu on StarPU buffer on CUDA
template<typename T>
void DGelu<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::dgelu::cuda<T>(stream, nelems, data);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

template<typename T>
void DGelu<std::tuple<T>>::submit(Index nelems, Handle data)
{
    Index *nelems_ = new Index{nelems};
    //double nflops = 5 * nelems;
    int ret = starpu_task_insert(&codelet,
            STARPU_RW, data.get(),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in dgelu task submission");
    }
}

//! Pack of dgelu operations for different types
dgelu_pack_t dgelu;

} // namespace nntile::starpu
