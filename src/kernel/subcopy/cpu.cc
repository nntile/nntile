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
    // using Y = typename CPUComputeType<T>::value;
    // auto src = reinterpret_cast<const Y *>(src_);
    // auto dst = reinterpret_cast<Y *>(dst_);
    using I = typename CPUComputeType<int64_t>::value;
    auto tmp_index = reinterpret_cast<I *>(tmp_index_);
    // Map temporary buffer into source index and destination index
    I *src_index = tmp_index;
    I *dst_index = tmp_index + ndim;
    // Get number of elements to copy and init source and target indexes for
    // the first element to copy
    Index nelems = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        nelems *= copy_shape[i];
        src_index[i] = src_start[i];
        dst_index[i] = dst_start[i];
    }
    // Get offsets for both source and target elements
    Index src_offset = src_start[0]; // src_stride[0] = 1
    Index dst_offset = dst_start[0]; // src_stride[0] = 1
    for(Index i = 1; i < ndim; ++i)
    {
        src_offset += src_start[i] * src_stride[i];
        dst_offset += dst_start[i] * dst_stride[i];
    }
    // Copy source into destination
    dst_[dst_offset] = src_[src_offset];
    // Get source and target offsets for the next element to copy
    ++src_offset;
    ++dst_offset;
    for(Index i = 1; i < nelems; ++i)
    {
        // Update indexes of source and target positions
        ++src_index[0];
        ++dst_index[0];
        // Get index and offset of the next source
        Index j = 0;
        while(src_index[j] == src_start[j]+copy_shape[j])
        {
            src_index[j] = src_start[j];
            ++j;
            ++src_index[j];
            src_offset += src_stride[j] - copy_shape[j-1]*src_stride[j-1];
        }
        // Get index and offset of the next target
        j = 0;
        while(dst_index[j] == dst_start[j]+copy_shape[j])
        {
            dst_index[j] = dst_start[j];
            ++j;
            ++dst_index[j];
            dst_offset += dst_stride[j] - copy_shape[j-1]*dst_stride[j-1];
        }
        // Copy source into destination
        dst_[dst_offset] = src_[src_offset];
        // Update offsets for the next copy
        ++src_offset;
        ++dst_offset;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp32_t *src, const Index *dst_start,
        const Index *dst_stride, fp32_t *dst, int64_t *tmp_index)
    noexcept;

template
void cpu<fp64_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const fp64_t *src, const Index *dst_start,
        const Index *dst_stride, fp64_t *dst, int64_t *tmp_index)
    noexcept;

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
void cpu<bf16_t>(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const bf16_t *src, const Index *dst_start,
        const Index *dst_stride, bf16_t *dst, int64_t *tmp_index)
    noexcept;

} // namespace nntile::kernel::subcopy
