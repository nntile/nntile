/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cuda/bias.cu
 * Bias operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/kernel/cuda/bias.cuh"

namespace nntile
{

template<typename T>
__global__
void bias_kernel_cuda(Index m, Index n, Index k, Index mk, const T *src,
        T *dst)
    noexcept
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            T *dst_slice = dst + i2*mk + i1;
            const T src_val = src[i2*m+i1];
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T &dst_val = dst_slice[i0*m];
                dst_val = dst_val + src_val;
            }
        }
    }
}

template<typename T>
void bias_starpu_cuda(void *buffers[], void *cl_args)
    noexcept
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *dst = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 blocks(16, 16), threads(8, 4);
    (bias_kernel_cuda<T>)<<<blocks, threads, 0, stream>>>(m, n, k, mk,
            src, dst);
}

// Expliciot instantiation
template
starpu_cuda_func_t bias_starpu_cuda<fp32_t>;

template
starpu_cuda_func_t bias_starpu_cuda<fp64_t>;

} // namespace nntile

