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

#ifndef STARPU_SIMGRID
#include "nntile/kernel/subcopy.hh"
#endif // STARPU_SIMGRID
#include "nntile/starpu/subcopy.hh"

namespace nntile::starpu::subcopy
{

//! Complex copying through StarPU buffers is available only on CPU
template<typename T>
void cpu(void *buffers[], void *cl_args)
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

//! Footprint for copy tasks that depend on copy shape
static
uint32_t footprint(struct starpu_task *task)
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

//Codelet codelet_fp16;
Codelet codelet_fp32, codelet_fp64, codelet_int64,
        codelet_bool, codelet_fp32_fast_tf32, codelet_bf16;

void init()
{
//    codelet_fp16.init("nntile_subcopy_fp16",
//            footprint,
//            {cpu<fp32_t>},
//            {}
//            );
    codelet_fp32.init("nntile_subcopy_fp32",
            footprint,
            {cpu<fp32_t>},
            {}
            );
    codelet_fp64.init("nntile_subcopy_fp64",
            footprint,
            {cpu<fp64_t>},
            {}
            );
    codelet_int64.init("nntile_subcopy_int64",
            footprint,
            {cpu<int64_t>},
            {}
            );
    codelet_bool.init("nntile_subcopy_bool",
            footprint,
            {cpu<bool_t>},
            {}
            );
    codelet_fp32_fast_tf32.init("nntile_subcopy_fp32_fast_tf32",
            footprint,
            {cpu<fp32_t>},
            {}
            );
    codelet_bf16.init("nntile_subcopy_bf16",
            footprint,
            {cpu<bf16_t>},
            {}
            );
}

void restrict_where(uint32_t where)
{
    //codelet_fp16.restrict_where(where);
    codelet_fp32.restrict_where(where);
    codelet_fp64.restrict_where(where);
    codelet_int64.restrict_where(where);
    codelet_bool.restrict_where(where);
    codelet_fp32_fast_tf32.restrict_where(where);
    codelet_bf16.restrict_where(where);
}

void restore_where()
{
    //codelet_fp16.restore_where();
    codelet_fp32.restore_where();
    codelet_fp64.restore_where();
    codelet_int64.restore_where();
    codelet_bool.restore_where();
    codelet_fp32_fast_tf32.restore_where();
    codelet_bf16.restore_where();
}

template<typename T>
void submit(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode)
{
    constexpr double nflops = 0;
    // Submit task
    int ret = starpu_task_insert(codelet<T>(),
            STARPU_VALUE, &(ndim), sizeof(ndim),
            STARPU_VALUE, &(src_start[0]), ndim*sizeof(src_start[0]),
            STARPU_VALUE, &(src_stride[0]), ndim*sizeof(src_stride[0]),
            STARPU_VALUE, &(copy_shape[0]), ndim*sizeof(copy_shape[0]),
            STARPU_VALUE, &(dst_start[0]), ndim*sizeof(dst_start[0]),
            STARPU_VALUE, &(dst_stride[0]), ndim*sizeof(dst_stride[0]),
            STARPU_R, static_cast<starpu_data_handle_t>(src),
            mode, static_cast<starpu_data_handle_t>(dst),
            STARPU_SCRATCH, static_cast<starpu_data_handle_t>(tmp_index),
            STARPU_FLOPS, nflops, // No floating point operations
            0);
    // Check submission
    if(ret != 0)
    {
        throw std::runtime_error("Error in subcopy task submission");
    }
}

// Explicit instantiation
//template
//void submit<fp16_t>(Index ndim, const std::vector<Index> &src_start,
//        const std::vector<Index> &src_stride,
//        const std::vector<Index> &dst_start,
//        const std::vector<Index> &dst_stride,
//        const std::vector<Index> &copy_shape, Handle src, Handle dst,
//        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<fp32_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<fp32_fast_tf32_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<fp64_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<int64_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<bool_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

template
void submit<bf16_t>(Index ndim, const std::vector<Index> &src_start,
        const std::vector<Index> &src_stride,
        const std::vector<Index> &dst_start,
        const std::vector<Index> &dst_stride,
        const std::vector<Index> &copy_shape, Handle src, Handle dst,
        Handle tmp_index, starpu_data_access_mode mode);

} // namespace nntile::starpu::subcopy
