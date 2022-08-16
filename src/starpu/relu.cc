/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/relu.cc
 * ReLU operation on a StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-16
 * */

#include "nntile/starpu/relu.hh"
#include "nntile/kernel/cpu/relu.hh"
#ifdef NNTILE_USE_CUDA
#   include "nntile/kernel/cuda/relu.hh"
#endif // NNTILE_USE_CUDA

namespace nntile
{
namespace starpu
{

//! Apply relu along middle axis of StarPU buffer on CPU
template<typename T>
void relu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    Index nelems = reinterpret_cast<Index *>(cl_args)[0];
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    // Launch kernel
    nntile::kernel::cpu::relu<T>(nelems, data);
}

#ifdef NNTILE_USE_CUDA
//! Apply relu along middle axis of StarPU buffer on CUDA
template<typename T>
void relu_cuda(void *buffers[], void *cl_args)
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
    nntile::kernel::cuda::relu<T>(stream, nelems, data);
}
#endif // NNTILE_USE_CUDA

StarpuCodelet relu_codelet_fp32("nntile_relu_fp32",
        nullptr,
        {relu_cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
        {relu_cuda<fp32_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

StarpuCodelet relu_codelet_fp64("nntile_relu_fp64",
        nullptr,
        {relu_cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
        {relu_cuda<fp64_t>}
#else // NNTILE_USE_CUDA
        {}
#endif // NNTILE_USE_CUDA
        );

void relu_restrict_where(uint32_t where)
{
    relu_codelet_fp32.restrict_where(where);
    relu_codelet_fp64.restrict_where(where);
}

void relu_restore_where()
{
    relu_codelet_fp32.restore_where();
    relu_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *relu_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *relu_codelet<fp32_t>()
{
    return &relu_codelet_fp32;
}

template<>
constexpr StarpuCodelet *relu_codelet<fp64_t>()
{
    return &relu_codelet_fp64;
}

template<typename T>
void relu(Index nelems, starpu_data_handle_t data)
{
    Index *nelems_ = new Index{nelems};
    //fp64_t nflops = 5 * nelems;
    int ret = starpu_task_insert(relu_codelet<T>(),
            STARPU_RW, data,
            STARPU_CL_ARGS, nelems_, sizeof(*nelems_),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in relu task submission");
    }
}

// Explicit instantiaion
template
void relu<fp32_t>(Index nelems, starpu_data_handle_t data);

template
void relu<fp64_t>(Index nelems, starpu_data_handle_t data);

} // namespace starpu
} // namespace nntile

