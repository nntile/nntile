/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/accumulate_hypot.cc
 * Accumulate one StarPU buffers into another as hypot
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/accumulate_hypot.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/hypot_inplace.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
AccumulateHypot<std::tuple<T>>::AccumulateHypot():
    codelet("nntile_accumulate_hypot", nullptr, cpu_funcs, cuda_funcs)
{
    // Modes cannot be variable for accumulate_hypot operation
    // Construct modes
    constexpr std::array<starpu_data_access_mode, 2> modes = {
        static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R
    };
    // Set modes
    codelet.set_modes_fixed(modes);
}

//! Apply accumulate_hypot operation for StarPU buffers in CPU
template<typename T>
void AccumulateHypot<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T);
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Launch kernel
    kernel::hypot_inplace::cpu<T>(nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AccumulateHypot<std::tuple<fp32_fast_tf32_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateHypot<std::tuple<fp32_fast_fp16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

template<>
void AccumulateHypot<std::tuple<fp32_fast_bf16_t>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cpu(buffers, cl_args);
}

#ifdef NNTILE_USE_CUDA
//! Apply accumulate_hypot for StarPU buffers on CUDA
template<typename T>
void AccumulateHypot<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    Index nelems = interfaces[0]->elemsize / sizeof(T);
    T *dst = interfaces[0]->get_ptr<T>();
    const T *src = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::hypot_inplace::cuda<T>(stream, nelems, 1.0, src, 1.0, dst);
#endif // STARPU_SIMGRID
}

// Specializations of CPU wrapper for accelerated types
template<>
void AccumulateHypot<std::tuple<fp32_fast_tf32_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateHypot<std::tuple<fp32_fast_fp16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}

template<>
void AccumulateHypot<std::tuple<fp32_fast_bf16_t>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Fall back to FP32
    AccumulateHypot<std::tuple<fp32_t>>::cuda(buffers, cl_args);
}
#endif // NNTILE_USE_CUDA

template<typename T>
void AccumulateHypot<std::tuple<T>>::submit(Handle src, Handle dst)
//! Insert accumulate hypoy task into StarPU pool of tasks
/*! No argument checking is performed. All the inputs are packed and passed to
 * starpu_task_insert() function. If task submission fails, this routines
 * throws an std::runtime_error() exception.
 * */
{
    //double nflops;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_RW | STARPU_COMMUTE, dst.get(),
            STARPU_R, src.get(),
            // STARPU_FLOPS, nflops,
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in accumulate_hypot task submission");
    }
}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class AccumulateHypot<std::tuple<nntile::fp64_t>>;
template class AccumulateHypot<std::tuple<nntile::fp32_t>>;
template class AccumulateHypot<std::tuple<nntile::fp32_fast_tf32_t>>;
template class AccumulateHypot<std::tuple<nntile::fp32_fast_fp16_t>>;
template class AccumulateHypot<std::tuple<nntile::fp32_fast_bf16_t>>;
template class AccumulateHypot<std::tuple<nntile::bf16_t>>;

//! Pack of accumulate_hypot operations for different types
accumulate_hypot_pack_t accumulate_hypot;

} // namespace nntile::starpu
