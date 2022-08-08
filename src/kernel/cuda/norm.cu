/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/norm.cu
 * Functions that compute different norms.
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#include "nntile/tile/norm.hh"
#include <cmath>

namespace nntile
{

static inline __device__
fp32_t cuda_abs(fp32_t x)
{
    return fabsf(x);
}

static inline __device__
fp64_t cuda_abs(fp64_t x)
{
    return fabs(x);
}

static inline __device__
fp32_t cuda_sqrt(fp32_t x)
{
    return sqrtf(x);
}

static inline __device__
fp64_t cuda_sqrt(fp64_t x)
{
    return sqrt(x);
}

// Compute sum and scaled sum of squares of a tile
template<typename T>
__global__ static
void cuda_sum_ssq_single_axis_init(Index m, Index n, Index k, Index mk,
        const T *src, T *sum_ssq)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            const T *src_slice = src + i2*mk + i1;
            T sum = 0, scale = 0, ssq = 0;
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update scale and scaled sum of squares
                if(val == 0)
                {
                    continue;
                }
                sum += val;
                T absval = cuda_abs(val);
                if(absval > scale)
                {
                    T tmp = scale / absval;
                    scale = absval;
                    ssq = ssq*tmp*tmp + T{1};
                }
                else
                {
                    T tmp = absval / scale;
                    ssq += tmp*tmp;
                }
            }
            Index dst_offset = 3 * (i2*m+i1);
            sum_ssq[dst_offset] = sum;
            sum_ssq[dst_offset+1] = scale;
            sum_ssq[dst_offset+2] = ssq;
        }
    }
}

template<typename T>
void norm_sum_ssq_codelet_cuda_single_axis_init(void *buffers[],
        void *cl_args)
{
    // Get sizes
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_sum_ssq_single_axis_init<T>)<<<blocks, threads, 0, stream>>>(m, n, k,
            mk, src, sum_ssq);
}

template
void norm_sum_ssq_codelet_cuda_single_axis_init<fp32_t>(void *buffers[],
        void *cl_args);

template
void norm_sum_ssq_codelet_cuda_single_axis_init<fp64_t>(void *buffers[],
        void *cl_args);

// Compute sum and scaled sum of squares of a tile
template<typename T>
__global__ static
void cuda_sum_ssq_single_axis_update(Index m, Index n, Index k, Index mk,
        const T *src, T *sum_ssq)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        for(Index i1 = i1_start; i1 < m; i1 += i1_step)
        {
            const T *src_slice = src + i2*mk + i1;
            Index dst_offset = 3 * (i2*m+i1);
            T &sum = sum_ssq[dst_offset];
            T &scale = sum_ssq[dst_offset+1];
            T &ssq = sum_ssq[dst_offset+2];
            for(Index i0 = 0; i0 < k; ++i0)
            {
                // Read value from source
                T val = src_slice[i0*m];
                // Update scale and scaled sum of squares
                if(val == 0)
                {
                    continue;
                }
                sum += val;
                T absval = cuda_abs(val);
                if(absval > scale)
                {
                    T tmp = scale / absval;
                    scale = absval;
                    ssq = ssq*tmp*tmp + T{1};
                }
                else
                {
                    T tmp = absval / scale;
                    ssq += tmp*tmp;
                }
            }
        }
    }
}

template<typename T>
void norm_sum_ssq_codelet_cuda_single_axis_update(void *buffers[],
        void *cl_args)
{
    // Get sizes
    Index m, n, k;
    starpu_codelet_unpack_args(cl_args, &m, &n, &k);
    const Index mk = m * k;
    // Get pointers
    const T *src = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *sum_ssq = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 blocks(16, 16), threads(8, 4);
    (cuda_sum_ssq_single_axis_update<T>)<<<blocks, threads, 0, stream>>>(m, n,
            k, mk, src, sum_ssq);
}

template
void norm_sum_ssq_codelet_cuda_single_axis_update<fp32_t>(void *buffers[],
        void *cl_args);

template
void norm_sum_ssq_codelet_cuda_single_axis_update<fp64_t>(void *buffers[],
        void *cl_args);

template<typename T>
__global__ static
void cuda_avg_dev_single_axis(Index m, T inv_nelems, T eps, const T *sum_ssq,
        T *avg_dev)
{
    Index i_start = threadIdx.x + blockIdx.x*blockDim.x,
          i_step = blockDim.x * gridDim.x;
    for(Index i = i_start; i < m; i += i_step)
    {
        const T avg = sum_ssq[3*i] * inv_nelems;
        T scale = sum_ssq[3*i+1];
        T scaled_avg_sqr = sum_ssq[3*i+2] * inv_nelems;
        avg_dev[2*i] = avg;
        // Mean of square values minus square of mean values
        // |avg| <= scale since |1/n sum x_i| <= max|x_i|
        T tmp = avg / scale;
        scaled_avg_sqr -= tmp * tmp;
        // Update by eps
        if(eps > 0)
        {
            if(scale >= eps)
            {
                T tmp = eps / scale;
                scaled_avg_sqr += tmp*tmp;
            }
            else
            {
                T tmp = scale / eps;
                scale = eps;
                scaled_avg_sqr *= tmp*tmp;
                scaled_avg_sqr += T{1};
            }
        }
        // Set deviation
        avg_dev[2*i+1] = scale * cuda_sqrt(scaled_avg_sqr);
    }
}

template<typename T>
void norm_avg_dev_codelet_cuda_single_axis(void *buffers[], void *cl_args)
{
    // Get sizes
    Index m, nelems;
    T eps;
    starpu_codelet_unpack_args(cl_args, &m, &nelems, &eps);
    const T inv_nelems = T{1} / static_cast<T>(nelems);
    // Get pointers
    const T *sum_ssq = reinterpret_cast<T *>(
            STARPU_VARIABLE_GET_PTR(buffers[0]));
    T *avg_dev = reinterpret_cast<T *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    cudaStream_t stream = starpu_cuda_get_local_stream();
    dim3 blocks(256), threads(32);
    (cuda_avg_dev_single_axis<T>)<<<blocks, threads, 0, stream>>>(m,
            inv_nelems, eps, sum_ssq, avg_dev);
}

template
void norm_avg_dev_codelet_cuda_single_axis<fp32_t>(void *buffers[],
        void *cl_args);

template
void norm_avg_dev_codelet_cuda_single_axis<fp64_t>(void *buffers[],
        void *cl_args);

} // namespace nntile

