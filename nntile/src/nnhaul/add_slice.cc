/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/add_slice.cc
 * StarPU wrappers for addition of a tensor and a broadcasted slice
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/add_slice.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_slice.hh"
#include "nntile/nnhaul/ops/add.hh"
#include "nntile/nnhaul/ops/scale.hh"
#include "nntile/nnhaul/ops/scale_slice.hh"

//! StarPU wrappers for add_slice operation
namespace nntile::haul
{

//! Constructor
template<typename T>
AddSlice<std::tuple<T>>::AddSlice():
    codelet("nntile_add_slice", &AddSlice<std::tuple<T>>::cpu, nullptr, &AddSlice<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::add_slice::cpu<T>
template<typename T>
void AddSlice<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    const T *src1 = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src2 = ::nntile::haul::buf_as<T>(buffers, 1);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 2);
    // Launch kernel
    kernel::add_slice::cpu<T>(
        args->m,
        args->n,
        args->k,
        args->alpha,
        src1,
        args->beta,
        src2,
        dst
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void AddSlice<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AddSlice<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AddSlice<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for add_slice tasks
template<typename T>
std::uint64_t AddSlice<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->m, sizeof(args->m), hash);
    hash = ::nntile::haul::fnv1a(&args->n, sizeof(args->n), hash);
    hash = ::nntile::haul::fnv1a(&args->k, sizeof(args->k), hash);
    return hash;
}

template<typename T>
void AddSlice<std::tuple<T>>::submit(int starpu_worker_hint,
    Index m,
    Index n,
    Index k,
    Scalar alpha,
    ::nnhaul::Handle & src1,
    Scalar beta,
    ::nnhaul::Handle & src2,
    ::nnhaul::Handle & dst
)
{
    // If k is 1, then this operation reduces to add
    if(k == 1)
    {
        add.submit<std::tuple<T>>(starpu_worker_hint, m*n, alpha, src1, beta, src2, dst);
        return;
    }
    // If alpha is zero then reduce to scale
    if(alpha == 0.0)
    {
        scale.submit<std::tuple<T>>(starpu_worker_hint, m * k * n, beta, src2, dst);
        return;
    }
    // If beta is zero then reduce to scale_slice
    if(beta == 0.0)
    {
        scale_slice.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, alpha, src1, dst);
        return;
    }
    // Access mode for the dst handle
    enum starpu_data_access_mode dst_mode;
    if(beta == 1.0)
    {
        dst_mode = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE);
    }
    else
    {
        dst_mode = STARPU_RW;
    }
    // Codelet arguments
    args_t *args = (args_t *)std::malloc(sizeof(*args));
    args->m = m;
    args->n = n;
    args->k = k;
    args->alpha = alpha;
    args->beta = beta;
    double nflops = m * n * (2*k+1);
    // Submit task
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &src1 }, { STARPU_R, &src2 }, { STARPU_W, &dst } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AddSlice<std::tuple<nntile::fp64_t>>;
template class AddSlice<std::tuple<nntile::fp32_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AddSlice<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AddSlice<std::tuple<nntile::bf16_t>>;
template class AddSlice<std::tuple<nntile::fp16_t>>;

//! Pack of add_slice operations for different types
add_slice_pack_t add_slice;

} // namespace nntile::haul
