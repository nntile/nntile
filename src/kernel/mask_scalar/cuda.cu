/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/mask_scalar/cuda.cu
 * Mask operation with scalar on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/mask_scalar/cuda.hh"

namespace nntile::kernel::mask_scalar
{

template<typename T>
static __global__
void cuda_kernel(Index nrows, Index ncols, const bool_t *mask, T val, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x,
        j = threadIdx.y + blockIdx.y*blockDim.y;
    if(i < nrows and j < ncols)
    {
        if(!mask[i])
        {
            data[j*nrows+i] = val;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nrows, Index ncols, const bool_t *mask,
        T val, T *data)
    noexcept
//! Set certain matrix entries to a given value by mask on CUDA
/*! Does the following operation:
 *      if(!mask[i]) data[i,:] = val
 *
 * @params[in] nrows: Number of rows of data
 * @params[in] ncols: Number of columns of data
 * @params[in] mask: buffer with mask values with nrows entries
 * @params[in] val: value to set if mask element is false
 * @params[in,out] data: nrows by ncols matrix, whose elements are updated
 * */
{
    dim3 blocks((nrows+255)/256, ncols), threads(256, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nrows, ncols, mask, val,
            data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, fp32_t val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, fp64_t val, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::mask_scalar
