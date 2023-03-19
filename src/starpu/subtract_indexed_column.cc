/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subtract_indexed_column.cc
 * Subtract a given value from the indexed column of a matrix for StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#include "nntile/starpu/subtract_indexed_column.hh"
#include "nntile/kernel/subtract_indexed_column.hh"

namespace nntile
{
namespace starpu
{
namespace subtract_indexed_column
{

//! Subtract given value from the indexed column in matrix stored in StarPU buffer on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto arg = reinterpret_cast<args_t<T>*>(cl_args);
    Index n_row = arg->n_row;
    T value = arg->value;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const Index* class_labels = interfaces[0]->get_ptr<Index>();
    T* dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::subtract_indexed_column::cpu<T>(n_row, value, class_labels, dst);
}

#ifdef NNTILE_USE_CUDA

#endif // NNTILE_USE_CUDA

Codelet codelet_fp32, codelet_fp64;

void init()
{
    codelet_fp32.init("nntile_subtract_indexed_column_fp32",
            nullptr,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            // {cuda<fp32_t>}
            {}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );
    codelet_fp64.init("nntile_subtract_indexed_column_fp64",
            nullptr,
            {cpu<fp64_t>},
#ifdef NNTILE_USE_CUDA
            // {cuda<fp64_t>}
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
void submit(Index n_row, T val, Handle class_labels, Handle dst)
{
    args_t<T>* arg = (args_t<T>*)malloc(sizeof(args_t<T>));
    arg->n_row = n_row;
    arg->value = val;
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(class_labels),
            STARPU_CL_ARGS, arg, sizeof(*arg),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            //STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in subtract_indexed_column task submission");
    }
}

// Explicit instantiation
template
void submit<fp32_t>(Index n_row, fp32_t val, Handle class_labels, Handle dst);

template
void submit<fp64_t>(Index n_row, fp64_t val, Handle class_labels, Handle dst);

} // namespace subtract_indexed_column
} // namespace starpu
} // namespace nntile