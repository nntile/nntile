/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate.cc
 * Accumulate one StarPU buffers into another
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/accumulate.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/add_inplace.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Accumulate<std::tuple<T>>::Accumulate():
    codelet("nntile_accumulate", &Accumulate<std::tuple<T>>::cpu, nullptr, nullptr)
{
    // Modes cannot be variable for accumulate operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 2> modes = {
        static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Apply accumulate operation for StarPU buffers in CPU
template<typename T>
void Accumulate<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get interfaces
    Index nelems = (*reinterpret_cast<std::size_t const *>(cl_args)) / sizeof(T);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 0);
    const T *src = ::nntile::haul::buf_as<T>(buffers, 1);
    // Launch kernel
    kernel::add_inplace::cpu<T>(nelems, 1.0, src, 1.0, dst);
}

// Specializations of CPU wrapper for accelerated types
template<>
void Accumulate<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Accumulate<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Accumulate<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Accumulate<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void Accumulate<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    Accumulate<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}


//! Submit accumulate task
template<typename T>
void Accumulate<std::tuple<T>>::submit(int starpu_worker_hint, ::nnhaul::Handle & src, ::nnhaul::Handle & dst)
//! Insert accumulate task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * nntile_starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    std::size_t nnhaul_nbytes = dst.nbytes();
    ::nntile::haul::insert_task(codelet.raw, starpu_worker_hint,
        { { STARPU_RW | STARPU_COMMUTE, &dst }, { STARPU_R, &src } }, &nnhaul_nbytes, sizeof(nnhaul_nbytes));

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Accumulate<std::tuple<nntile::fp64_t>>;
template class Accumulate<std::tuple<nntile::fp32_t>>;
template class Accumulate<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Accumulate<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Accumulate<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Accumulate<std::tuple<nntile::bf16_t>>;

//! Pack of accumulate operations for different types
accumulate_pack_t accumulate;

} // namespace nntile::haul
