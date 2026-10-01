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
#include "nntile/nnhaul/ops/subtract_indexed_outputs.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subtract_indexed_outputs.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
SubtractIndexedOutputs<std::tuple<T>>::SubtractIndexedOutputs():
    codelet(
        "nntile_subtract_indexed_outputs",
        &SubtractIndexedOutputs<std::tuple<T>>::cpu,
#ifdef NNTILE_USE_CUDA
        &SubtractIndexedOutputs<std::tuple<T>>::cuda,
#else
        nullptr,
#endif
        &SubtractIndexedOutputs<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    Scalar val = args->value;
    // Get interfaces
    const int64_t *labels = ::nntile::haul::buf_as<int64_t>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::subtract_indexed_outputs::cpu<T>(n_labels, n_outputs, ignore_index,
        val, labels, dst);
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
void SubtractIndexedOutputs<std::tuple<T>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t*>(cl_args);
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    const int64_t *labels = ::nntile::haul::buf_as<int64_t>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    // Get CUDA stream
    cudaStream_t stream = ::nnhaul::cuda_stream();
    // Launch kernel
    kernel::subtract_indexed_outputs::cuda<T>(stream, n_labels, n_outputs,
            ignore_index, args->value, labels, dst);
}

// Specializations of CUDA wrapper for accelerated types
template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_tf32_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_fp16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void SubtractIndexedOutputs<std::tuple<fp32_fast_bf16_t>>::cuda(
    void *buffers[],
    void *cl_args)
    noexcept
{
    // Fall back to FP32
    SubtractIndexedOutputs<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

//! Footprint for subtract_indexed_outputs tasks
template<typename T>
std::uint64_t SubtractIndexedOutputs<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Apply hash over parameters m, n and k
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = ::nntile::haul::fnv1a(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

template<typename T>
void SubtractIndexedOutputs<std::tuple<T>>::submit(
        Index n_labels, Index n_outputs, Index ignore_index,
            Scalar val, ::nnhaul::Handle & labels, ::nnhaul::Handle & dst)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(args_t));
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->value = val;
    args->ignore_index = ignore_index;
    ::nntile::haul::insert_task(
        codelet.raw,
        -1,
        {{STARPU_R, &labels}, {STARPU_RW, &dst}},
        args,
        sizeof(*args));
    std::free(args);
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

} // namespace nntile::haul
