/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/subcopy.cc
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/nnhaul/ops/subcopy.hh"

// Standard libraries
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subcopy.hh"
#include "nntile/nnhaul/pack_args.hh"

namespace nntile::haul
{

//! Constructor
template<typename T>
Subcopy<std::tuple<T>>::Subcopy():
    codelet("nntile_subcopy", &Subcopy<std::tuple<T>>::cpu, nullptr, &Subcopy<std::tuple<T>>::footprint)
{
    // Modes are not fixed, they are decided during runtime by default
}

//! Complex copying through StarPU buffers on CPU
template<typename T>
void Subcopy<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    const T *src = ::nntile::haul::buf_as<T>(buffers, 0);
    T *dst = ::nntile::haul::buf_as<T>(buffers, 1);
    std::vector<int64_t> tmp_index_storage(static_cast<std::size_t>(*ndim_ptr > 0 ? *ndim_ptr : 1));
    int64_t *tmp_index = tmp_index_storage.data();
    // Launch kernel
    kernel::subcopy::cpu<T>(*ndim_ptr, src_start, src_stride,
            copy_shape, src, dst_start, dst_stride, dst, tmp_index);
}


//! Footprint for subcopy tasks that depend on copy shape
template<typename T>
std::uint64_t Subcopy<std::tuple<T>>::footprint(void const *cl_args, std::size_t) noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    ::nntile::haul::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    std::size_t copy_shape_size = *ndim_ptr * sizeof(*copy_shape);
    // Apply hash over parameter copy_shape
    return ::nntile::haul::fnv1a(copy_shape, copy_shape_size, 0);
}

template<typename T>
void Subcopy<std::tuple<T>>::submit(int starpu_worker_hint,
        Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, ::nnhaul::Handle & src, ::nnhaul::Handle & dst,
        ::nnhaul::Handle & tmp_index, starpu_data_access_mode mode)
{
    constexpr double nflops = 0;
    // Submit task
    ::nntile::haul::insert_values(codelet.raw, starpu_worker_hint,
        { { &(ndim), sizeof(ndim) }, { &(src_start[0]), ndim*sizeof(src_start[0]) }, { &(src_stride[0]), ndim*sizeof(src_stride[0]) }, { &(copy_shape[0]), ndim*sizeof(copy_shape[0]) }, { &(dst_start[0]), ndim*sizeof(dst_start[0]) }, { &(dst_stride[0]), ndim*sizeof(dst_stride[0]) } },
        { { STARPU_R, &src }, { mode, &dst } });

}

// Explicit instantiation
// For some strange reason, the compiler does not instantiate the template
// automatically, so we need to do it manually
template class Subcopy<std::tuple<nntile::int64_t>>;
template class Subcopy<std::tuple<nntile::bool_t>>;
template class Subcopy<std::tuple<nntile::fp64_t>>;
template class Subcopy<std::tuple<nntile::fp32_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_tf32_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_fp16_t>>;
template class Subcopy<std::tuple<nntile::fp32_fast_bf16_t>>;
template class Subcopy<std::tuple<nntile::bf16_t>>;
template class Subcopy<std::tuple<nntile::fp16_t>>;

//! Pack of subcopy operations for different types
subcopy_pack_t subcopy;

} // namespace nntile::haul
