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
#include "nntile/nnhaul/ops/total_sum_accum.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/total_sum_accum.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
TotalSumAccum<std::tuple<T>>::TotalSumAccum():
    codelet("nntile_total_sum_accum", &TotalSumAccum<std::tuple<T>>::cpu, nullptr, &TotalSumAccum<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::total_sum_accum::cpu<T>
template<typename T>
void TotalSumAccum<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    Scalar alpha = args->alpha;
    Index n_labels = args->n_labels;
    Index n_outputs = args->n_outputs;
    Index ignore_index = args->ignore_index;
    // Get interfaces
    const T *logsumexp = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src = ::nntile::haul::buf_as<T>(buffers, 1);
    const int64_t* labels = ::nntile::haul::buf_as<int64_t>(buffers, 2);
    float *val = ::nntile::haul::buf_as<float>(buffers, 3);
    // Launch kernel
    kernel::total_sum_accum::cpu<T>(alpha, n_labels, n_outputs, ignore_index, logsumexp, src,
            labels, val);
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


//! Footprint for total_sum_accum tasks
template<typename T>
std::uint64_t TotalSumAccum<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->n_labels, sizeof(args->n_labels),
            hash);
    hash = ::nntile::haul::fnv1a(&args->n_outputs, sizeof(args->n_outputs),
            hash);
    return hash;
}

template<typename T>
void TotalSumAccum<std::tuple<T>>::submit(int starpu_worker_hint, Scalar alpha, Index n_labels,
        Index n_outputs, Index ignore_index,
            ::nnhaul::Handle & logsumexp, ::nnhaul::Handle & src, ::nnhaul::Handle & class_labels, ::nnhaul::Handle & val)
{
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->alpha = alpha;
    args->n_labels = n_labels;
    args->n_outputs = n_outputs;
    args->ignore_index = ignore_index;
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &logsumexp }, { STARPU_R, &src }, { STARPU_R, &class_labels }, { STARPU_RW | STARPU_COMMUTE, &val } }, args, sizeof(*args));
    std::free(args);

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
template class TotalSumAccum<std::tuple<nntile::fp16_t>>;

//! Pack of total_sum_accum operations for different types
total_sum_accum_pack_t total_sum_accum;

} // namespace nntile::haul
