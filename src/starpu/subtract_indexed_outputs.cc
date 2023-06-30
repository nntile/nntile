/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subtract_indexed_outputs.cc
 * Subtract a given value from certain matrix elements for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-29
 * */

#include "nntile/starpu/subtract_indexed_outputs.hh"
#include "nntile/kernel/subtract_indexed_outputs.hh"

namespace nntile
{
namespace starpu
{
namespace subtract_indexed_outputs
{

template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T>*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    T value = args->value;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const Index* labels = interfaces[0]->get_ptr<Index>();
    T* dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::subtract_indexed_outputs::cpu<T>(n_labels, n_outputs, value,
            labels, dst);
}

#ifdef NNTILE_USE_CUDA
//! Apply subtract_indexed_outputs operation on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t<T>*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    T value = args->value;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const Index* labels = interfaces[0]->get_ptr<Index>();
    T* dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::subtract_indexed_outputs::cuda<T>(stream, n_labels, n_outputs, value,
            labels, dst);
}
#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_subtract_indexed_outputs_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_subtract_indexed_outputs_fp64",
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
void submit(Index n_labels, Index n_outputs, T val, Handle labels, Handle dst)
{
    // Codelet arguments
    args_t<T>* args = (args_t<T>*)malloc(sizeof(args_t<T>));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->value = val;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(labels),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in subtract_indexed_outputs task "
                "submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index n_labels, Index n_outputs, fp32_t val, Handle labels,
        Handle dst);

template
void submit<fp64_t>(Index n_labels, Index n_outputs, fp64_t val, Handle labels,
        Handle dst);

} // namespace subtract_indexed_outputs
} // namespace starpu
} // namespace nntile
