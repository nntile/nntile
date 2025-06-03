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
#include "nntile/starpu/subcopy.hh"

// Standard libraries
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/kernel/subcopy.hh"
#include "nntile/starpu/config.hh"

namespace nntile::starpu
{

//! Constructor
template<typename T>
Subcopy<std::tuple<T>>::Subcopy():
    codelet("nntile_subcopy", footprint, cpu_funcs, cuda_funcs)
{
}

//! Complex copying through StarPU buffers on CPU
template<typename T>
void Subcopy<std::tuple<T>>::cpu(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    Config::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    int64_t *tmp_index = interfaces[2]->get_ptr<int64_t>();
    // Launch kernel
    kernel::subcopy::cpu<T>(*ndim_ptr, src_start, src_stride,
            copy_shape, src, dst_start, dst_stride, dst, tmp_index);
#endif // STARPU_SIMGRID
}

#ifdef NNTILE_USE_CUDA
//! Complex copying through StarPU buffers on CUDA
template<typename T>
void Subcopy<std::tuple<T>>::cuda(void *buffers[], void *cl_args)
    noexcept
{
#ifndef STARPU_SIMGRID // Run the code only if this is not a simulation
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    Config::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    auto interfaces = reinterpret_cast<VariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Launch kernel
    kernel::subcopy::cuda<T>(stream, *ndim_ptr, src_start, src_stride,
            copy_shape, src, dst_start, dst_stride, dst);
#endif // STARPU_SIMGRID
}
#endif // NNTILE_USE_CUDA

//! Footprint for subcopy tasks that depend on copy shape
template<typename T>
uint32_t Subcopy<std::tuple<T>>::footprint(struct starpu_task *task)
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    Config::unpack_args_ptr(task->cl_arg, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    std::size_t copy_shape_size = *ndim_ptr * sizeof(*copy_shape);
    // Apply hash over parameter copy_shape
    return starpu_hash_crc32c_be_n(copy_shape, copy_shape_size, 0);
}

template<typename T>
void Subcopy<std::tuple<T>>::submit(
        Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode)
{
    constexpr double nflops = 0;
    // Submit task
    int ret = starpu_task_insert(&codelet,
            STARPU_VALUE, &(ndim), sizeof(ndim),
            STARPU_VALUE, &(src_start[0]), ndim*sizeof(src_start[0]),
            STARPU_VALUE, &(src_stride[0]), ndim*sizeof(src_stride[0]),
            STARPU_VALUE, &(copy_shape[0]), ndim*sizeof(copy_shape[0]),
            STARPU_VALUE, &(dst_start[0]), ndim*sizeof(dst_start[0]),
            STARPU_VALUE, &(dst_stride[0]), ndim*sizeof(dst_stride[0]),
            STARPU_R, src.get(),
            mode, dst.get(),
            STARPU_SCRATCH, tmp_index.get(),
            STARPU_FLOPS, nflops, // No floating point operations
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in subcopy task submission");
    }
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

//! Pack of subcopy operations for different types
subcopy_pack_t subcopy;

} // namespace nntile::starpu
