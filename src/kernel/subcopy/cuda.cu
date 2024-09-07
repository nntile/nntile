/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subcopy/cuda.cu
 * Subcopy operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/subcopy/cuda.hh"
#include <array>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::subcopy
{

template<int NDIM>
struct packarg
{
    Index value[NDIM];
};

template<typename T, int NDIM>
static __global__
void cuda_kernel(Index nelems, packarg<NDIM> src_start,
        packarg<NDIM> src_stride, packarg<NDIM> copy_shape, const T *src,
        packarg<NDIM> dst_start, packarg<NDIM> dst_stride, T *dst)
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
 * */
{
    Index i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        Index src_offset=0, dst_offset=0;
        for(int idim = 0; idim < NDIM; ++idim)
        {
            Index j = i % copy_shape.value[idim];
            i = i / copy_shape.value[idim];
            src_offset += src_stride.value[idim] * (j+src_start.value[idim]);
            dst_offset += dst_stride.value[idim] * (j+dst_start.value[idim]);
        }
        dst[dst_offset] = src[src_offset];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const T *src,
        const Index *dst_start, const Index *dst_stride, T *dst)
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
 * */
{
    Index nelems = 1;
    for(Index i = 0; i < ndim; ++i)
    {
        nelems *= copy_shape[i];
    }
    dim3 threads(256);
    dim3 blocks((nelems+255)/256);
    switch(ndim)
    {
        case 1:
            {
                packarg<1> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 1; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        case 2:
            {
                packarg<2> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 2; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        case 3:
            {
                packarg<3> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 3; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        case 4:
            {
                packarg<4> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 4; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        case 5:
            {
                packarg<5> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 5; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        case 6:
            {
                packarg<6> src_start_, src_stride_, copy_shape_, dst_start_,
                    dst_stride_;
                for(int i = 0; i < 6; i++)
                {
                    src_start_.value[i] = src_start[i];
                    src_stride_.value[i] = src_stride[i];
                    copy_shape_.value[i] = copy_shape[i];
                    dst_start_.value[i] = dst_start[i];
                    dst_stride_.value[i] = dst_stride[i];
                }
                (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
                        src_start_, src_stride_, copy_shape_, src,
                        dst_start_, dst_stride_, dst);
                break;
            }
        default:
            fprintf(stderr, "SUBCOPY of unsupported ndim\n");
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const fp32_t *src_,
        const Index *dst_start, const Index *dst_stride, fp32_t *dst_)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const fp64_t *src_,
        const Index *dst_start, const Index *dst_stride, fp64_t *dst_)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const bf16_t *src_,
        const Index *dst_start, const Index *dst_stride, bf16_t *dst_)
    noexcept;

template
void cuda<int64_t>(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const int64_t *src_,
        const Index *dst_start, const Index *dst_stride, int64_t *dst_)
    noexcept;

template
void cuda<bool_t>(cudaStream_t stream, Index ndim, const Index *src_start,
        const Index *src_stride, const Index *copy_shape, const bool_t *src_,
        const Index *dst_start, const Index *dst_stride, bool_t *dst_)
    noexcept;

} // namespace nntile::kernel::subcopy
