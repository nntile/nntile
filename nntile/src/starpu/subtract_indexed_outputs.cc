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

// Corresponding header
#include "nntile/starpu/subtract_indexed_outputs.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subtract_indexed_outputs.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
SubtractIndexedOutputs<std::tuple<T>>::SubtractIndexedOutputs():
    codelet("nntile_subtract_indexed_outputs", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
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

// Specializations of CPU wrapper for accelerated types
template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply subtract_indexed_outputs operation on StarPU buffer on CUDA
template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
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

// Specializations of CUDA wrapper for accelerated types
template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for subtract_indexed_outputs tasks
template<typename T>
uint32_t SubtractIndexedOutputs<std::tuple<T>>::footprint(struct starpu_task *task)
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

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::submit(
        Index n_labels, Index n_outputs, Index ignore_index,
            Scalar val, Handle labels, Handle dst)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(args_t));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->value = val;
    args->ignore_index = ignore_index;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, labels.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW, dst.get(),
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
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class SubtractIndexedOutputs<std::tuple<nntile::fp64_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_tf32_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_fp16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp32_fast_bf16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::bf16_t>>;
template class SubtractIndexedOutputs<std::tuple<nntile::fp16_t>>;

//! Pack of subtract_indexed_outputs operations for different types
subtract_indexed_outputs_pack_t subtract_indexed_outputs;

} // namespace nntile::starpu
