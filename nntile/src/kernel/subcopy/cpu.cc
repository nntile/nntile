/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subcopy/cpu.cc
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/subcopy/cpu.hh"
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::subcopy
{

namespace
{

template<typename I>
I linear_offset(Index ndim, const I *index, const Index *stride)
{
    I offset = 0;
    for(Index i = 0; i < ndim; ++i)
    {
        offset += index[i] * stride[i];
    }
    return offset;
}

template<typename I>
void advance_c_order_index(Index ndim, const Index *start, const Index *shape,
    I *index)
{
    if(ndim == 0)
    {
        return;
    }
    Index k = ndim - 1;
    ++index[k];
    while(k >= 0 && index[k] == start[k] + shape[k])
    {
        index[k] = start[k];
        if(k == 0)
        {
            break;
        }
        --k;
        ++index[k];
    }
}

} // namespace

template<typename T>
void cpu(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const T *src_, const Index *dst_start,
        const Index *dst_stride, T *dst_, int64_t *tmp_index_)
    noexcept
//! Complex copying of one multidimensional array into another
/*! This function is not meant for a performant implementation, as its sole
 * purpose is an easy data redistribution. It helps, for example, in case of
 * converting between a single contiguous array on a single node (e.g., a
 * Python numpy or torch array) and a distributed allocation on many nodes
 * (e.g., nntile data distribution).
 * A simple memory copy shall be treated with a help of starpu_data_cpy()
 * function.
 *
 * @param[in] ndim: Dimensionality of underlying arrays
 * @param[in] src_start: Start element to copy from source array. Contains ndim
 *      values.
 * @param[in] src_stride: Strides of the source array. Contains ndim values.
 * @param[in] copy_shape: Shape of array to copy. Contains ndim values.
 * @param[in] src_: Pointer to input data
 * @param[in] dst_start: Start element to copy to destination array. Contains
 *      ndim values.
 * @param[in] dst_stride: Strides of the destination array. Contains ndim
 *      values.
 * @param[inout] dst_: Pointer to output data
 * @param[out] tmp_index_: Temporary buffer for indexing. Contains 2*ndim
 *      values.
 * */
{
    using I = typename CPUComputeType<int64_t>::value;
    auto tmp_index = reinterpret_cast<I *>(tmp_index_);
    I *src_index = tmp_index;
    I *dst_index = tmp_index + ndim;
    Index nelems = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        nelems *= copy_shape[i];
        src_index[i] = src_start[i];
        dst_index[i] = dst_start[i];
    }
    for(Index elem = 0; elem < nelems; ++elem)
    {
        const Index src_offset =
            linear_offset(ndim, src_index, src_stride);
        const Index dst_offset =
            linear_offset(ndim, dst_index, dst_stride);
        dst_[dst_offset] = src_[src_offset];
        if(elem + 1 < nelems)
        {
            advance_c_order_index(ndim, src_start, copy_shape, src_index);
            advance_c_order_index(ndim, dst_start, copy_shape, dst_index);
        }
    }
}

// Explicit instantiation
template
void cpu<int64_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const nntile::int64_t *src, const Index *dst_start,
        const Index *dst_stride, nntile::int64_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<bool_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const bool_t *src, const Index *dst_start,
        const Index *dst_stride, bool_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp32_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp32_t *src, const Index *dst_start,
        const Index *dst_stride, fp32_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp32_fast_tf32_t *src, const Index *dst_start,
        const Index *dst_stride, fp32_fast_tf32_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp32_fast_fp16_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp32_fast_fp16_t *src, const Index *dst_start,
        const Index *dst_stride, fp32_fast_fp16_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp32_fast_bf16_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp32_fast_bf16_t *src, const Index *dst_start,
        const Index *dst_stride, fp32_fast_bf16_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp64_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp64_t *src, const Index *dst_start,
        const Index *dst_stride, fp64_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<bf16_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const bf16_t *src, const Index *dst_start,
        const Index *dst_stride, bf16_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp16_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp16_t *src, const Index *dst_start,
        const Index *dst_stride, fp16_t *dst, int64_t *tmp_index)
    noexcept;

} // namespace nntile::kernel::subcopy
