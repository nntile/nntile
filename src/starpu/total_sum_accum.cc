/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/total_sum_accum.cc
 * Total sum accumulating for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-16
 * */

#include "nntile/starpu/total_sum_accum.hh"
#include "nntile/kernel/total_sum_accum.hh"

namespace nntile
{
namespace starpu
{
namespace total_sum_accum
{

//! total_sum_accumulation operation of StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *logsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    const Index* labels = interfaces[2]->get_ptr<Index>();
    T* val = interfaces[3]->get_ptr<T>();
    // Launch kernel
    kernel::total_sum_accum::cpu<T>(n_labels, n_outputs, logsumexp, src,
            labels, val);
}

#ifdef NNTILE_USE_CUDA
//! Apply total_sum_accum operation on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *logsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    const Index* labels = interfaces[2]->get_ptr<Index>();
    T* val = interfaces[3]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::total_sum_accum::cuda<T>(stream, n_labels, n_outputs, logsumexp, src,
            labels, val);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_total_sum_accum_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_total_sum_accum_fp64",
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
void submit(Index n_labels, Index n_outputs, Handle logsumexp, Handle src,
        Handle class_labels, Handle val)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(logsumexp),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            STARPU_R, static_cast<starpu_data_handle_t>(class_labels),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, static_cast<starpu_data_handle_t>(val),
            //Config::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(val),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in total_sum_accum task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index n_labels, Index n_outputs, Handle logsumexp,
        Handle src, Handle class_labels, Handle val);

template
void submit<fp64_t>(Index n_labels, Index n_outputs, Handle logsumexp,
        Handle src, Handle class_labels, Handle val);

} // namespace total_sum_accum
} // namespace starpu
} // namespace nntile
