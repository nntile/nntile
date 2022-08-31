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
 * @date 2022-08-31
 * */

#include "nntile/starpu/gelutanh.hh"
#include "nntile/kernel/cpu/gelutanh.hh"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/gelutanh.hh"
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace starpu
{

//! Apply approximate gelu along middle axis of StarPU buffer
template<typename T>
void gelutanh_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    nntile::kernel::cpu::gelutanh<T>(nelems, data);
}

#ifdef NNTILE_USE_CUDA
//! Apply approximate gelu along middle axis of StarPU buffer
template<typename T>
void gelutanh_cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    nntile::kernel::cuda::gelutanh<T>(stream, nelems, data);
}
#endif // NNTILE_USE_CUDA

StarpuCodelet gelutanh_codelet_fp32, gelutanh_codelet_fp64;
StarpuCodelet gelutanh_codelet_fp32("nntile_gelutanh_fp32",
        nullptr,
        {gelutanh_cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
        {gelutanh_cuda<fp32_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

StarpuCodelet gelutanh_codelet_fp64("nntile_gelutanh_fp64",
        nullptr,
        {gelutanh_cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
        {gelutanh_cuda<fp64_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

void gelutanh_restrict_where(uint32_t where)
{
    gelutanh_codelet_fp32.restrict_where(where);
    gelutanh_codelet_fp64.restrict_where(where);
}

void gelutanh_restore_where()
{
    gelutanh_codelet_fp32.restore_where();
    gelutanh_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *gelutanh_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *gelutanh_codelet<fp32_t>()
{
    return &gelutanh_codelet_fp32;
}

template<>
constexpr StarpuCodelet *gelutanh_codelet<fp64_t>()
{
    return &gelutanh_codelet_fp64;
}

template<typename T>
void gelutanh(Index nelems, starpu_data_handle_t data)
{
    Index *nelems_ = new Index{nelems};
    //fp64_t nflops = 5 * nelems;
    int ret = starpu_task_insert(gelutanh_codelet<T>(),
            STARPU_RW, data,
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
void gelutanh<fp32_t>(Index nelems, starpu_data_handle_t data);

template
void gelutanh<fp64_t>(Index nelems, starpu_data_handle_t data);

} // namespace starpu
} // namespace nntile

