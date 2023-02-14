/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/addcdiv.cc
 * Per-element addcdiv operation of StarPU buffers
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-14
 * */

#include "nntile/starpu/addcdiv.hh"
#include "nntile/kernel/addcdiv.hh"

namespace nntile
{
namespace starpu
{
//! StarPU wrappers for addcdiv operation
namespace addcdiv
{

//! Apply addcdiv operation on StarPU buffers on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T> *>(cl_args);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *nom = interfaces[0]->get_ptr<T>();
    const T *denom = interfaces[1]->get_ptr<T>();
    T *src = interfaces[2]->get_ptr<T>();
    // Launch kernel
    kernel::addcdiv::cpu<T>(args->val, args->eps, args->nelems, nom, denom, src);
}

#ifdef NNTILE_USE_CUDA
//! Apply addcdiv on StarPU buffer on CUDA
// template<typename T>
// void cuda(void *buffers[], void *cl_args)
//     noexcept
// {
//     // // Get arguments
//     // Index nelems = reinterpret_cast<Index *>(cl_args)[0];
//     // // Get interfaces
//     // auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
//     // const T *src = interfaces[0]->get_ptr<T>();
//     // T *dst = interfaces[1]->get_ptr<T>();
//     // // Get CUDA stream
//     // cudaStream_t stream = starpu_cuda_get_local_stream();
//     // // Launch kernel
//     // kernel::prod::cuda<T>(stream, nelems, src, dst);
// }
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_addcdiv_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_addcdiv_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            {}
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
void submit(T val, T eps, Index nelems, Handle nom, Handle denom, Handle src)
{
    args_t<T>* args = (args_t<T>*)malloc(sizeof(args_t<T>));
    args->val = val;
    args->eps = eps;
    args->nelems = nelems;
    //fp64_t nflops = 5 * nelems;
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
void submit<fp32_t>(fp32_t val, fp32_t eps, Index nelems, Handle nom, Handle denom, Handle src);

template
void submit<fp64_t>(fp64_t val, fp64_t eps, Index nelems, Handle nom, Handle denom, Handle src);

} // namespace addcdiv
} // namespace starpu
} // namespace nntile