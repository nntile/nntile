/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/starpu/copy.cc
 * Smart copy StarPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-11
 * */

#include "nntile/starpu/copy.hh"
#include "nntile/kernel/cpu/copy.hh"

namespace nntile
{
namespace starpu
{

//! Smart copying through StarPU buffers
template<typename T>
void copy_cpu(void *buffers[], void *cl_args)
    noexcept
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    Starpu::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    // Get interfaces
    auto interfaces = reinterpret_cast<StarpuVariableInterface **>(buffers);
    const T *src = interfaces[0]->get_ptr<T>();
    T *dst = interfaces[1]->get_ptr<T>();
    Index *tmp_index = interfaces[2]->get_ptr<Index>();
    // Launch kernel
    nntile::kernel::cpu::copy<T>(*ndim_ptr, src_start, src_stride, copy_shape,
            src, dst_start, dst_stride, dst, tmp_index);
}

//! Footprint for copy tasks that depend on copy shape
static
uint32_t copy_footprint(struct starpu_task *task)
{
    // Get arguments
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    Starpu::unpack_args_ptr(task->cl_arg, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    std::size_t copy_shape_size = *ndim_ptr * sizeof(*copy_shape);
    // Apply hash over parameter copy_shape
    return starpu_hash_crc32c_be_n(copy_shape, copy_shape_size, 0);
}

StarpuCodelet copy_codelet_fp32("nntile_copy_fp32",
        copy_footprint,
        {copy_cpu<fp32_t>},
        {}
        );

StarpuCodelet copy_codelet_fp64("nntile_copy_fp64",
        copy_footprint,
        {copy_cpu<fp64_t>},
        {}
        );

void copy_restrict_where(uint32_t where)
{
    copy_codelet_fp32.restrict_where(where);
    copy_codelet_fp64.restrict_where(where);
}

void copy_restore_where()
{
    copy_codelet_fp32.restore_where();
    copy_codelet_fp64.restore_where();
}

template<typename T>
constexpr StarpuCodelet *copy_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *copy_codelet<fp32_t>()
{
    return &copy_codelet_fp32;
}

template<>
constexpr StarpuCodelet *copy_codelet<fp64_t>()
{
    return &copy_codelet_fp64;
}

template<typename T>
void copy(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape,
        starpu_data_handle_t src, starpu_data_handle_t dst,
        starpu_data_handle_t tmp_index, starpu_data_access_mode mode)
{
    constexpr fp64_t zero_flops = 0;
    // Submit task
    int ret = starpu_task_insert(copy_codelet<T>(),
            STARPU_VALUE, &(ndim), sizeof(ndim),
            STARPU_VALUE, &(src_start[0]), ndim*sizeof(src_start[0]),
            STARPU_VALUE, &(src_stride[0]), ndim*sizeof(src_stride[0]),
            STARPU_VALUE, &(copy_shape[0]), ndim*sizeof(copy_shape[0]),
            STARPU_VALUE, &(dst_start[0]), ndim*sizeof(dst_start[0]),
            STARPU_VALUE, &(dst_stride[0]), ndim*sizeof(dst_stride[0]),
            STARPU_R, src,
            mode, dst,
            STARPU_SCRATCH, tmp_index,
            STARPU_FLOPS, zero_flops, // No floating point operations
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in copy task submission");
    }
}

// Explicit instantiation
template
void copy<fp32_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape,
        starpu_data_handle_t src, starpu_data_handle_t dst,
        starpu_data_handle_t tmp_index, starpu_data_access_mode mode);

template
void copy<fp64_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape,
        starpu_data_handle_t src, starpu_data_handle_t dst,
        starpu_data_handle_t tmp_index, starpu_data_access_mode mode);

} // namespace starpu
} // namespace nntile

