/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/gelutanh.cc
 * Approximate GeLU operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-25
 * */

#include "nntile/starpu/gelutanh.hh"
#include "nntile/kernel/gelutanh.hh"

namespace nntile
{
namespace starpu
{
namespace gelutanh
{

//! Apply approximate gelu on StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    kernel::gelutanh::cpu<T>(nelems, data);
}

#ifdef NNTILE_USE_CUDA
//! Apply approximate gelu on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::gelutanh::cuda<T>(stream, nelems, data);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_gelutanh_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_gelutanh_fp64",
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
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index nelems, Handle data)
{
    Index *nelems_ = new Index{nelems};
    //fp64_t nflops = 5 * nelems;
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_RW, static_cast<starpu_data_handle_t>(data),
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in gelutanh task submission");
    }
}

// Explicit instantiaion
template
void submit<fp32_t>(Index nelems, Handle data);

template
void submit<fp64_t>(Index nelems, Handle data);

} // namespace gelutanh
} // namespace starpu
} // namespace nntile

