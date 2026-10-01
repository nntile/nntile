/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/adam_step.cc
 * Fused Adam step operation of StarPU buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/adam_step.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/adam_step.hh"

//! StarPU wrappers for one step of Adam optimizer
namespace nntile::haul
{

//! Constructor
template<typename T>
AdamStep<std::tuple<T>>::AdamStep():
    codelet("nntile_adam_step", &AdamStep<std::tuple<T>>::cpu, nullptr, &AdamStep<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Apply Adam step on StarPU buffers on CPU
template<typename T>
void AdamStep<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    // Get interfaces
    T *grad = ::nntile::haul::buf_as<T>(buffers, 0);
    T *first_moments = ::nntile::haul::buf_as<T>(buffers, 1);
    T *second_moments = ::nntile::haul::buf_as<T>(buffers, 2);
    T* p = ::nntile::haul::buf_as<T>(buffers, 3);
    // Launch kernel
    kernel::adam_step::cpu<T>(
        args->num_iter,
        args->num_elems,
        args->beta_1,
        args->beta_2,
        args->eps,
        args->lr,
        args->weight_decay,
        grad,
        first_moments,
        second_moments,
        p
    );
}

// Specializations of CPU wrapper for accelerated types
template<>
void AdamStep<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AdamStep<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AdamStep<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Footprint for adam_step tasks that depends only on cl_arg
template<typename T>
std::uint64_t AdamStep<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    auto args = reinterpret_cast<args_t const *>(cl_args);
    uint32_t hash = 0;
    hash = ::nntile::haul::fnv1a(&args->num_elems, sizeof(args->num_elems), hash);
    return hash;
}

//! Submit Adam step task
template<typename T>
void AdamStep<std::tuple<T>>::submit(int starpu_worker_hint,
    Index num_iter,
    Index num_elems,
    Scalar beta_1,
    Scalar beta_2,
    Scalar eps,
    Scalar lr,
    Scalar weight_decay,
    ::nnhaul::Handle & grad,
    ::nnhaul::Handle & first_moment,
    ::nnhaul::Handle & second_moment,
    ::nnhaul::Handle & param
)
{
    // Codelet arguments
    args_t* args = (args_t*)std::malloc(sizeof(*args));
    args->num_iter = num_iter;
    args->num_elems = num_elems;
    args->beta_1 = beta_1;
    args->beta_2 = beta_2;
    args->eps = eps;
    args->lr = lr;
    args->weight_decay = weight_decay;
    //double nflops = 5 * nelems;
    // Submit task
    enum starpu_data_access_mode moments_mode;
    if (num_iter == 1)
    {
        moments_mode = STARPU_W;
    }
    else
    {
        moments_mode = STARPU_RW;
    }
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_R, &grad }, { moments_mode, &first_moment }, { moments_mode, &second_moment }, { STARPU_RW, &param } }, args, sizeof(*args));
    std::free(args);

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AdamStep<std::tuple<nntile::fp64_t>>;
template class AdamStep<std::tuple<nntile::fp32_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AdamStep<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AdamStep<std::tuple<nntile::bf16_t>>;
template class AdamStep<std::tuple<nntile::fp16_t>>;

//! Pack of adam_step operations for different types
adam_step_pack_t adam_step;

} // namespace nntile::haul
