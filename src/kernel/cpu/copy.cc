/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/copy.cc
 * Smart copy operation
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cpu/copy.hh"
#include "nntile/starpu.hh"

namespace nntile
{

//! Smart copying
template<typename T>
void copy_kernel_cpu(Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const T *src,
        const Index *dst_start, const Index *dst_stride, T *dst,
        Index *tmp_index)
    noexcept
{
    Index *src_index = tmp_index;
    Index *dst_index = tmp_index + ndim;
    Index nelems = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        nelems *= copy_shape[i];
        src_index[i] = src_start[i];
        dst_index[i] = dst_start[i];
    }
    Index src_offset = src_start[0]; // src_stride[0] = 1
    Index dst_offset = dst_start[0]; // src_stride[0] = 1
    for(Index i = 1; i < ndim; ++i)
    {
        src_offset += src_start[i] * src_stride[i];
        dst_offset += dst_start[i] * dst_stride[i];
    }
    dst[dst_offset] = src[src_offset];
    ++src_offset;
    ++dst_offset;
    for(Index i = 1; i < nelems; ++i)
    {
        ++src_index[0];
        ++dst_index[0];
        Index j = 0;
        while(src_index[j] == src_start[j]+copy_shape[j])
        {
            src_index[j] = src_start[j];
            ++j;
            ++src_index[j];
            src_offset += src_stride[j] - copy_shape[j-1]*src_stride[j-1];
        }
        j = 0;
        while(dst_index[j] == dst_start[j]+copy_shape[j])
        {
            dst_index[j] = dst_start[j];
            ++j;
            ++dst_index[j];
            dst_offset += dst_stride[j] - copy_shape[j-1]*dst_stride[j-1];
        }
        dst[dst_offset] = src[src_offset];
        ++src_offset;
        ++dst_offset;
    }
}

// Smart copying through StarPU buffers
template<typename T>
void copy_starpu_cpu(void *buffers[], void *cl_args)
    noexcept
{
    const Index *ndim_ptr, *src_start, *src_stride, *copy_shape, *dst_start,
          *dst_stride;
    // Read arguments
    Starpu::unpack_args_ptr(cl_args, ndim_ptr, src_start, src_stride,
            copy_shape, dst_start, dst_stride);
    Index ndim = *ndim_ptr;
    const T *src = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_NDIM_GET_PTR(buffers[1]));
    Index *tmp_index = reinterpret_cast<Index *>(STARPU_NDIM_GET_PTR(
                buffers[2]));
    copy_kernel_cpu<T>(ndim, src_start, src_stride, copy_shape, src, dst_start,
            dst_stride, dst, tmp_index);
}

// Explicit instantiation
template
void copy_starpu_cpu<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void copy_starpu_cpu<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

