/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subtract_indexed_outputs.cc
 * Subtract a given value from certain matrix elements for StarPU buffer
 *
 * @version 1.1.0
 * */

#ifndef STARPU_SIMGRID
#include "nntile/kernel/subtract_indexed_outputs.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/subtract_indexed_outputs.hh"

namespace nntile::starpu::subtract_indexed_outputs
{

template<typename T>
void cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    Scalar val = args->value;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *labels = interfaces[0]->get_ptr<int64_t>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::subtract_indexed_outputs::cpu<T>(n_labels, n_outputs, ignore_index,
        val, labels, dst);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Apply subtract_indexed_outputs operation on StarPU buffer on CUDA
template<typename T>
void cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const int64_t *labels = interfaces[0]->get_ptr<int64_t>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::subtract_indexed_outputs::cuda<T>(stream, n_labels, n_outputs,
            ignore_index, args->value, labels, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for subtract_indexed_outputs tasks
static
uint32_t footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = starpu_hash_crc32c_be_n(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

Codelet codelet_fp32, codelet_fp64, codelet_fp32_fast_tf32, codelet_bf16,
        codelet_fp32_fast_fp16, codelet_fp32_fast_bf16;

void init()
{
    codelet_fp32.init("nntile_subtract_indexed_outputs_fp32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_bf16.init("nntile_subtract_indexed_outputs_bf16",
            footprint,
            {cpu<bf16_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<bf16_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_tf32.init("nntile_subtract_indexed_outputs_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_fp16.init("nntile_subtract_indexed_outputs_fp32_fast_fp16",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp32_fast_bf16.init("nntile_subtract_indexed_outputs_fp32_fast_bf16",
            footprint,
            {cpu<fp32_t>},
#ifdef NNTILE_USE_CUDA
            {cuda<fp32_t>}
#else // NNTILE_USE_CUDA
            {}
#endif // NNTILE_USE_CUDA
            );

    codelet_fp64.init("nntile_subtract_indexed_outputs_fp64",
            footprint,
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
    codelet_bf16.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_fp32_fast_fp16.restrict_where(where);
    codelet_fp32_fast_bf16.restrict_where(where);
    codelet_fp64.restrict_where(where);
}

void restore_where()
{
    codelet_fp32.restore_where();
    codelet_bf16.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_fp32_fast_fp16.restore_where();
    codelet_fp32_fast_bf16.restore_where();
    codelet_fp64.restore_where();
}

template<typename T>
void submit(Index n_labels, Index n_outputs, Index ignore_index,
            Scalar val, Handle labels, Handle dst)
{
    // Codelet arguments
    args_t* args = (args_t*)malloc(sizeof(args_t));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->value = val;
    args->ignore_index = ignore_index;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_R, static_cast<starpu_data_handle_t>(labels),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, static_cast<starpu_data_handle_t>(dst),
            //Config::STARPU_RW_COMMUTE, static_cast<starpu_data_handle_t>(dst),
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
void submit<fp32_t>(Index n_labels, Index n_outputs, Index ignore_index,
                    Scalar val, Handle labels, Handle dst);

template
void submit<bf16_t>(Index n_labels, Index n_outputs, Index ignore_index,
                    Scalar val, Handle labels, Handle dst);

template
void submit<fp32_fast_tf32_t>(Index n_labels, Index n_outputs, Index ignore_index,
                              Scalar val, Handle labels, Handle dst);

template
void submit<fp32_fast_fp16_t>(Index n_labels, Index n_outputs, Index ignore_index,
                              Scalar val, Handle labels, Handle dst);

template
void submit<fp32_fast_bf16_t>(Index n_labels, Index n_outputs, Index ignore_index,
                              Scalar val, Handle labels, Handle dst);

template
void submit<fp64_t>(Index n_labels, Index n_outputs, Index ignore_index,
                    Scalar val, Handle labels, Handle dst);

} // namespace nntile::starpu::subtract_indexed_outputs
