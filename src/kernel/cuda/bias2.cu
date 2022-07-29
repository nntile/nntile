/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cuda/bias2.cu
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cuda/bias2.cuh"

namespace nntile
{

template<typename T>
__global__
void bias2_kernel_cuda(Index m, Index n, Index k, const T *avg_dev,
        T *dst)
    noexcept
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    // Outer loop by the last mode of source and destination tiles
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Middle loop by the middle mode of destination tile
        for(Index i1 = i1_start; i1 < k; i1 += i1_step)
        {
            Index src_offset = 2 * m * i2;
            Index dst_offset = (i2*k+i1) * m;
            // Inner loop by the first mode of source and destination tiles
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T &val = dst[dst_offset];
                // Corresponding mean and deviation
                const T &avg = avg_dev[src_offset];
                const T &dev = avg_dev[src_offset+1];
                // Normalization
                val = (val-avg) / dev;
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

// CUDA codelet for normalization over single axis
template<typename T>
void bias2_starpu_cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Source (avg_dev) is a 2-by-m-by-n tile, which contains mean and
    // deviation values
    // Destination is an m-by-k-by-n tile
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const T *avg_dev = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_bias_avg_dev_single_axis<T>)<<<blocks, threads, 0, stream>>>(m, n, k,
            avg_dev, dst);
}

// Explicit instantiation
template
void bias2_starpu_cuda<fp32_t>(void *buffers[], void *cl_args)
    noexcept;

template
void bias2_starpu_cuda_<fp64_t>(void *buffers[], void *cl_args)
    noexcept;

} // namespace nntile

