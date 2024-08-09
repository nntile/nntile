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

#ifndef STARPU_SIMGRID
#include "nntile/kernel/accumulate_maxsumexp.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/accumulate_maxsumexp.hh"
#include <cstdlib>

//! StarPU wrappers for accumulate_maxsumexp operation
namespace nntile::starpu::accumulate_maxsumexp
{

//! Apply accumulate_maxsumexp operation for StarPU buffers in CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
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
void cuda(void *buffers[], void *cl_args)
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

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
    codelet_fp32.init("nntile_accumulate_maxsumexp_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp32.nbuffers = 2;
    codelet_fp32.modes[0] = static_cast<starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    codelet_fp32.modes[1] = STARPU_R;

    codelet_fp32_fast_tf32.init("nntile_accumulate_maxsumexp_fp32_fast_tf32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp32_fast_tf32.nbuffers = 2;
    codelet_fp32_fast_tf32.modes[0] = static_cast<starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    codelet_fp32_fast_tf32.modes[1] = STARPU_R;


    codelet_fp64.init("nntile_accumulate_maxsumexp_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp64_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.nbuffers = 2;
    codelet_fp64.modes[0] = static_cast<starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    codelet_fp64.modes[1] = STARPU_R;

    codelet_bf16.init("nntile_accumulate_maxsumexp_bf16",
            nullptr,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_bf16.nbuffers = 2;
    codelet_bf16.modes[0] = static_cast<starpu_data_access_mode>(
            STARPU_RW | STARPU_COMMUTE);
    codelet_bf16.modes[1] = STARPU_R;
}

void restrict_where(uint32_t where)
{
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_bf16.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp64.restore_where();
    codelet_fp32_fast_tf32.restore_where();
}

template<typename T>
void submit(Handle src, Handle dst)
//! Insert accumulate_maxsumexp task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW|STARPU_COMMUTE, static_cast<starpu_data_handle_t>(dst),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in accumulate_maxsumexp task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Handle src, Handle dst);

template
void submit<fp32_fast_tf32_t>(Handle src, Handle dst);

template
void submit<fp64_t>(Handle src, Handle dst);

template
void submit<bf16_t>(Handle src, Handle dst);

} // namespace nntile::starpu::accumulate_maxsumexp
