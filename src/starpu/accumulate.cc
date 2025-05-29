/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate.cc
 * Accumulate one StarPU buffers into another
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/accumulate.hh"

// Standard libraries
#include <cstdlib>

// Other NNTile headers
#include "nntile/kernel/add_inplace.hh"

//! StarPU wrappers for accumulate operation
namespace nntile::starpu::accumulate
{

//! Apply accumulate operation for StarPU buffers in CPU
template<typename T>
void KernelWrapper<T>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T);
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::add_inplace::cpu<T>(nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply accumulate for StarPU buffers on CUDA
template<typename T>
void KernelWrapper<T>::cuda(void *buffers[], void *cl_args)
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
    kernel::add_inplace::cuda<T>(stream, nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Define codelet pack
codelet_pack_t codelet_pack = codelet_pack_t(
    "nntile_accumulate",
    nullptr
).set_modes({
    static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
    STARPU_R
});

// void restrict_where(uint32_t where)
// {
//     codelet_fp64.restrict_where(where);
//     codelet_fp32.restrict_where(where);
//     codelet_fp32_fast_tf32.restrict_where(where);
//     codelet_fp32_fast_fp16.restrict_where(where);
//     codelet_fp32_fast_bf16.restrict_where(where);
//     codelet_bf16.restrict_where(where);
// }

// void restore_where()
// {
//     codelet_fp64.restore_where();
//     codelet_fp32.restore_where();
//     codelet_fp32_fast_tf32.restore_where();
//     codelet_fp32_fast_bf16.restore_where();
//     codelet_fp32_fast_fp16.restore_where();
//     codelet_bf16.restore_where();
// }

template<typename T>
void submit(Handle src, Handle dst)
//! Insert accumulate task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    int ret = starpu_task_insert(codelet_pack.get_codelet<T>(),
            STARPU_RW | STARPU_COMMUTE, dst.get(),
            STARPU_R, src.get(),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in accumulate task submission");
    }
}

// Explicit instantiation
template
void submit<fp64_t>(Handle src, Handle dst);

template
void submit<fp32_t>(Handle src, Handle dst);

template
void submit<fp32_fast_tf32_t>(Handle src, Handle dst);

template
void submit<fp32_fast_fp16_t>(Handle src, Handle dst);

template
void submit<fp32_fast_bf16_t>(Handle src, Handle dst);

template
void submit<bf16_t>(Handle src, Handle dst);

} // namespace nntile::starpu::accumulate
