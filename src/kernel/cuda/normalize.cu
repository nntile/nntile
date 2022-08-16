/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cuda/normalize.cu
 * Normalize operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-15
 * */

#include "nntile/kernel/cuda/normalize.hh"

namespace nntile
{
namespace kernel
{
namespace cuda
{

template<typename T>
static __global__
void normalize_kernel(Index m, Index n, Index k, Index l, T eps,
        const T *gamma, const T *beta, const T *sumnorm, T *dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T one = 1;
    const T invl = one / T(l);
    const T rinvl = sqrt(invl);
    const T reps = sqrt(eps);
    // Outer loop by the last mode of dst and sumnorm arrays
    for(Index i2 = i2_start; i2 < n; i2 += i2_step)
    {
        // Middle loop by the middle mode of dst array
        for(Index i1 = i1_start; i1 < k; i1 += i1_step)
        {
            Index src_offset = 2 * m * i2;
            Index dst_offset = (i2*k+i1) * m;
            // Inner loop by the first mode of dst and sumnorm arrays
            for(Index i0 = 0; i0 < m; ++i0)
            {
                // Value-to-update
                T &val = dst[dst_offset];
                // Corresponding mean and root-mean-square
                const T sum = sumnorm[src_offset];
                const T mean = sum * invl;
                const T norm = sumnorm[src_offset+1];
                const T rms = norm * rinvl;
                // Deviation
                T dev;
                if(rms > reps)
                {
                    T tmp = mean/rms, tmp2 = reps/rms;
                    T ssq = one - tmp*tmp;
                    ssq += tmp2*tmp2;
                    dev = rms * sqrt(ssq);
                }
                else
                {
                    T tmp = rms/reps, tmp2 = mean/reps;
                    T ssq = tmp*tmp - tmp2*tmp2;
                    ssq += one;
                    dev = reps * sqrt(ssq);
                }
                // Normalization
                val = (val-mean) / dev;
                // Renormalization
                val = val*gamma[0] + beta[0];
                // Update pointers
                ++dst_offset;
                src_offset += 2;
            }
        }
    }
}

template<typename T>
void normalize(cudaStream_t stream, Index m, Index n, Index k, Index l, T eps,
        const T *gamma, const T *beta, const T *sumnorm, T *dst)
    noexcept
//! Renormalize buffer along middle axis
/*! Provided m-by-k-by-n output array dst is renormalized along second axis
 * with k elements. The following operations is applied:
 *      dst[i, :, j] := (dst[i, :, j]-mean(i, j)) / sqrt(var(i, j)+eps)
 *          * gamma + beta
 * where mean and var functions are computed as follows:
 *      mean(i, j) = sumnorm[0, i, j] / l
 *      var(i, j) = sumnorm[1, i, j]^2/l - mean(i,j)^2
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] l: Number of elements used to calculate sum and Euclidian norm
 * @param[in] eps: Regularization parameter for variance
 * @param[in] gamma: Deviation for the renormalized output
 * @param[in] beta: Mean value for the renormalized output
 * @param[in] sumnorm: Sums and norms of slices
 * @param[in] dst: Contiguous output array
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    (normalize_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, l, eps,
            gamma, beta, sumnorm, dst);
}

// Explicit instantiation
template
void normalize<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index l,
        fp32_t eps, const fp32_t *gamma, const fp32_t *beta,
        const fp32_t *sumnorm, fp32_t *dst)
    noexcept;

template
void normalize<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index l,
        fp64_t eps, const fp64_t *gamma, const fp64_t *beta,
        const fp64_t *sumnorm, fp64_t *dst)
    noexcept;

} // namespace cuda
} // namespace kernel
} // namespace nntile

