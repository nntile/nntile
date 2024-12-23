/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelu.cc
 * GeLU operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/gelu.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/gelu.hh"

//! StarPU wrappers for GeLU operation
namespace nntile::starpu::gelu
{

//! Apply gelu on StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::gelu::cpu<T>(nelems, data);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply gelu on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
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
    kernel::gelu::cuda<T>(stream, nelems, data);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64, codelet_bf16,
        codelet_fp32_fast_tf32, codelet_fp32_fast_bf16, codelet_fp32_fast_fp16;

void init()
{
    codelet_fp32.init("nntile_gelu_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_gelu_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_bf16.init("nntile_gelu_fp32_fast_bf16",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_fp16.init("nntile_gelu_fp32_fast_fp16",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_gelu_fp32_fast_tf32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_gelu_bf16",
            nullptr,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_bf16.restrict_where(where);
    codelet_fp32_fast_fp16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_bf16.restore_where();
    codelet_fp32_fast_fp16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
}

template<typename T>
void submit(Index nelems, Handle data)
{
    Index *nelems_ = new Index{nelems};
    //double nflops = 5 * nelems;
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(data),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelu task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Index nelems, Handle data);

template
void submit<fp64_t>(Index nelems, Handle data);

template
void submit<bf16_t>(Index nelems, Handle data);

template
void submit<fp32_fast_bf16_t>(Index nelems, Handle data);

template
void submit<fp32_fast_fp16_t>(Index nelems, Handle data);

template
void submit<fp32_fast_tf32_t>(Index nelems, Handle data);

} // namespace nntile::starpu::gelu
