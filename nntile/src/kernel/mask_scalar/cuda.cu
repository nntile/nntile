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
 * @version 1.1.0
 * */

#include "nntile/kernel/mask_scalar/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::mask_scalar
{

template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP>
static __global__
void cuda_kernel2(Index nrows, Index ncols, const bool *mask, Scalar val_,
        T *data)
{
    Index griddim_row = (nrows+BLOCK_ROW-1) / BLOCK_ROW;
    Index block_i = blockIdx.x % griddim_row;
    Index block_j = blockIdx.x / griddim_row;
    Index i = threadIdx.x % BLOCK_ROW;
    Index j = threadIdx.x / BLOCK_ROW;
    Index global_i = block_i*BLOCK_ROW + i;
    using Y = typename T::repr_t;
    const T val = static_cast<T>(static_cast<Y>(val_));
    __shared__ bool mask_block[BLOCK_ROW];
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    if(threadIdx.x < BLOCK_ROW and global_i < nrows)
    {
        mask_block[threadIdx.x] = mask[global_i];
    }
    __syncthreads();
    if(global_i < nrows)
    {
        if(!mask[global_i])
        {
            for(Index global_j = block_j*BLOCK_COL+j;
                    global_j < ::min((block_j+1)*BLOCK_COL, ncols);
                    global_j += BLOCK_COL_STEP)
            {
                data[global_j*nrows+global_i] = val;
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nrows, Index ncols, const bool_t *mask_,
        Scalar val, T *data)
    noexcept
//! Set certain matrix entries to a given value by mask on CUDA
/*! Does the following operation:
 *      if(!mask[i]) data[i,:] = val
 *
 * @params[in] nrows: Number of rows of data
 * @params[in] ncols: Number of columns of data
 * @params[in] mask_: buffer with mask values with nrows entries
 * @params[in] val: value to set if mask element is false
 * @params[inout] data: nrows by ncols matrix, whose elements are updated
 * */
{
    dim3 threads(256);
    using B = typename CUDAComputeType<bool_t>::value;
    auto mask = reinterpret_cast<const B *>(mask_);
    dim3 blocks(((nrows+127)/128) * ((ncols+31)/32));
    (cuda_kernel2<T, 128, 32, 16>)<<<blocks, threads, 0, stream>>>(nrows,
            ncols, mask, val, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nrows, Index ncols,
        const bool_t *mask, Scalar val, fp16_t *data)
    noexcept;

} // namespace nntile::kernel::mask_scalar
