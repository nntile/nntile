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
#include "nntile/nnhaul/ops/randn.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/randn.hh"
#include "nntile/nnhaul/pack_args.hh"


namespace nntile::haul
{

//! Constructor
template<typename T>
Randn<std::tuple<T>>::Randn():
    codelet("nntile_randn", &Randn<std::tuple<T>>::cpu, nullptr, &Randn<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! StarPU wrapper for kernel::randn::cpu<T>
template<typename T>
void Randn<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr, mean_ptr,
            stddev_ptr, start, shape, stride, underlying_shape);
    // Get interfaces
    Index ndim = *ndim_ptr;
    T *data = ::nntile::haul::buf_as<T>(buffers, 0);
    // Index walk writes one past ndim; StarPU scratch is 2 * ndim.
    std::size_t const scratch_n =
        static_cast<std::size_t>(*ndim_ptr > 0 ? *ndim_ptr : 1);
    std::vector<nntile::int64_t> tmp_index_storage(scratch_n * 2);
    nntile::int64_t *tmp_index = tmp_index_storage.data();
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
}


//! Footprint for randn tasks that depend on shape
template<typename T>
std::uint64_t Randn<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    const Index *ndim_ptr, *nelems_ptr, *start, *shape, *stride,
          *underlying_shape;
    const unsigned long long *seed_ptr;
    const Scalar *mean_ptr, *stddev_ptr;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, nelems_ptr, seed_ptr,
            mean_ptr, stddev_ptr, start, shape, stride, underlying_shape);
    std::size_t shape_size = *ndim_ptr * sizeof(*shape);
    // Apply hash over parameter copy_shape
    return ::nntile::haul::fnv1a(shape, shape_size, 0);
}

template<typename T>
void Randn<std::tuple<T>>::submit(int starpu_worker_hint,
    Index ndim,
    Index nelems,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev,
    const std::vector<Index> &start,
    const std::vector<Index> &shape,
    const std::vector<Index> &stride,
    const std::vector<Index> &underlying_shape,
    ::nnhaul::Handle & data
)
{
    double nflops = 2 * nelems;
    // Submit task
    ::nntile::haul::insert_values(codelet.raw, starpu_worker_hint,
        { { &ndim, sizeof(ndim) }, { &nelems, sizeof(nelems) }, { &seed, sizeof(seed) }, { &mean, sizeof(mean) }, { &stddev, sizeof(stddev) }, { &start[0], ndim*sizeof(start[0]) }, { &shape[0], ndim*sizeof(shape[0]) }, { &stride[0], ndim*sizeof(stride[0]) }, { &underlying_shape[0], ndim*sizeof(underlying_shape[0]) } },
        { { STARPU_W, &data } });

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

} // namespace nntile::haul
