/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/total_sum_accum.cc
 * Total sum accumulating for StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/total_sum_accum.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/total_sum_accum.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
TotalSumAccum<std::tuple<T>>::TotalSumAccum():
    codelet("nntile_total_sum_accum", footprint, cpu_funcs, cuda_funcs)
{
}

//! StarPU wrapper for kernel::total_sum_accum::cpu<T>
template<typename T>
void TotalSumAccum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    Scalar alpha = args->alpha;
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *logsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    const int64_t* labels = interfaces[2]->get_ptr<int64_t>();
    float *val = interfaces[3]->get_ptr<float>();
    // Launch kernel
    kernel::total_sum_accum::cpu<T>(alpha, n_labels, n_outputs, ignore_index, logsumexp, src,
            labels, val);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void TotalSumAccum<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! StarPU wrapper for kernel::total_sum_accum::cuda<T>
template<typename T>
void TotalSumAccum<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    auto args = reinterpret_cast<args_t *>(cl_args);
    Scalar alpha = args->alpha;
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *logsumexp = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    const int64_t* labels = interfaces[2]->get_ptr<int64_t>();
    float *val = interfaces[3]->get_ptr<float>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::total_sum_accum::cuda<T>(stream, alpha, n_labels, n_outputs,
        ignore_index, logsumexp, src, labels, val);
#endif // STARPU_SIMGRID
}

// Specializations of CUDA wrapper for accelerated types
template<>
void TotalSumAccum<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void TotalSumAccum<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    TotalSumAccum<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for total_sum_accum tasks
template<typename T>
uint32_t TotalSumAccum<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    auto args = reinterpret_cast<args_t *>(task->cl_arg);
    uint32_t hash = 0;
    hash = starpu_hash_crc32c_be_n(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = starpu_hash_crc32c_be_n(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

template<typename T>
void TotalSumAccum<std::tuple<T>>::submit(Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index,
            Handle logsumexp, Handle src, Handle class_labels, Handle val)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->alpha = alpha;
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->ignore_index = ignore_index;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_R, logsumexp.get(),
            STARPU_R, src.get(),
            STARPU_R, class_labels.get(),
            STARPU_CL_ARGS, args, sizeof(*args),
            STARPU_RW | STARPU_COMMUTE, val.get(),
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in total_sum_accum task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class TotalSumAccum<std::tuple<nntile::fp64_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_tf32_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_fp16_t>>;
template class TotalSumAccum<std::tuple<nntile::fp32_fast_bf16_t>>;
template class TotalSumAccum<std::tuple<nntile::bf16_t>>;

//! Pack of total_sum_accum operations for different types
total_sum_accum_pack_t total_sum_accum;

} // namespace nntile::starpu
