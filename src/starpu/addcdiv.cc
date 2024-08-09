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

#ifndef STARPU_SIMGRID
#include "nntile/kernel/addcdiv.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/addcdiv.hh"
#include <cstdlib>

//! StarPU wrappers for addcdiv operation
namespace nntile::starpu::addcdiv
{

//! Apply addcdiv operation on StarPU buffers on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
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
    kernel::addcdiv::cpu<T>(args->val, args->eps, args->nelems, nom, denom,
            src);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply addcdiv on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
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
    kernel::addcdiv::cuda<T>(stream, args->val, args->eps, args->nelems, nom,
            denom, src);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_addcdiv_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_addcdiv_bf16",
            nullptr,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_addcdiv_fp32_fast_tf32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_addcdiv_fp64",
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
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Scalar val, Scalar eps, Index nelems, Handle nom, Handle denom, Handle src)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->val = val;
    args->eps = eps;
    args->nelems = nelems;
    //double nflops = 5 * nelems;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(nom),
            STARPU_R, static_cast<starpu_data_handle_t>(denom),
            STARPU_RW, static_cast<starpu_data_handle_t>(src),
            STARPU_CL_ARGS, args, sizeof(*args),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in addcdiv task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Scalar val, Scalar eps, Index nelems, Handle nom,
        Handle denom, Handle src);

template
void submit<bf16_t>(Scalar val, Scalar eps, Index nelems, Handle nom,
        Handle denom, Handle src);

template
void submit<fp32_fast_tf32_t>(Scalar val, Scalar eps, Index nelems, Handle nom,
        Handle denom, Handle src);

template
void submit<fp64_t>(Scalar val, Scalar eps, Index nelems, Handle nom,
        Handle denom, Handle src);

} // namespace nntile::starpu::addcdiv
