/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/randn.cc
 * Randn operation on a StarPU buffer
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/starpu/randn.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/randn.hh"
#include "nntile/starpu/config.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Randn<std::tuple<T>>::Randn():
    codelet("nntile_randn", footprint, cpu_funcs, cuda_funcs)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::randn::cpu<T>
template<typename T>
void Randn<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    Config::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr, mean_ptr,
            stddev_ptr, start, shape, stride, underlying_shape);
    // Get interfaces
    Index ndim = *ndim_ptr;
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    T *data = interfaces[0]->get_ptr<T>();
    nntile::int64_t *tmp_index = interfaces[1]->get_ptr<nntile::int64_t>();
    // Launch kernel
    kernel::randn::cpu<T>(
        ndim,
        *nelems_ptr,
        *seed_ptr,
        *mean_ptr,
        *stddev_ptr,
        start,
        shape,
        underlying_shape,
        data,
        stride,
        tmp_index
    );
#endif // STARPU_SIMGRID
}

//! Footprint for randn tasks that depend on shape
template<typename T>
uint32_t Randn<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    Config::unpack_args_ptr(task->cl_arg, ndim_ptr, nelems_ptr, seed_ptr,
            mean_ptr, stddev_ptr, start, shape, stride, underlying_shape);
    std::size_t shape_size = *ndim_ptr * sizeof(*shape);
    // Apply hash over parameter copy_shape
    return starpu_hash_crc32c_be_n(shape, shape_size, 0);
}

template<typename T>
void Randn<std::tuple<T>>::submit(
    Index ndim,
    Index nelems,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev,
    const std::vector<Index> &start,
    const std::vector<Index> &shape,
    const std::vector<Index> &stride,
    const std::vector<Index> &underlying_shape,
    Handle data,
    Handle tmp_index
)
{
    double nflops = 2 * nelems;
    // Submit task
    int ret = starpu_task_insert(&codelet,
        STARPU_VALUE, &ndim, sizeof(ndim),
        STARPU_VALUE, &nelems, sizeof(nelems),
        STARPU_VALUE, &seed, sizeof(seed),
        STARPU_VALUE, &mean, sizeof(mean),
        STARPU_VALUE, &stddev, sizeof(stddev),
        STARPU_VALUE, &start[0], ndim*sizeof(start[0]),
        STARPU_VALUE, &shape[0], ndim*sizeof(shape[0]),
        STARPU_VALUE, &stride[0], ndim*sizeof(stride[0]),
        STARPU_VALUE, &underlying_shape[0],
        ndim*sizeof(underlying_shape[0]),
        STARPU_W, data.get(),
        STARPU_SCRATCH, tmp_index.get(),
        STARPU_FLOPS, nflops,
        0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in randn task submission");
    }
}
// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Randn<std::tuple<nntile::fp64_t>>;
template class Randn<std::tuple<nntile::fp32_t>>;
template class Randn<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Randn<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Randn<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Randn<std::tuple<nntile::bf16_t>>;

//! Pack of randn operations for different types
randn_pack_t randn;

} // namespace nntile::starpu
