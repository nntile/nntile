/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/normalize/cuda.cu
 * Normalize operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/normalize/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::normalize
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index size, T eps,
        const T *gamma, const T *beta, const T *sumnorm, T *dst)
{
    Index i2_start = threadIdx.x + blockIdx.x*blockDim.x,
          i1_start = threadIdx.y + blockIdx.y*blockDim.y,
          i2_step = blockDim.x * gridDim.x,
          i1_step = blockDim.y * gridDim.y;
    constexpr T one = 1;
    const T invl = one / T(size);
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
                // Deviation=sqrt(rms*rms-mean*mean+reps*reps)
                T dev;
                // Although in theory tmp<=1 it is not always true in practice
                // due presence of rounding errors
                T tmp = ::fabs(mean) / rms;
                // Check if rounding errors broke theoretical invariant
                if(tmp >= one)
                {
                    dev = reps;
                }
                else if(rms > reps)
                {
                    T ssq = one - tmp*tmp;
                    T tmp2 = reps / rms;
                    ssq += tmp2*tmp2;
                    dev = rms * ::sqrt(ssq);
                }
                else
                {
                    T ssq = one - tmp*tmp;
                    T tmp2 = rms / reps;
                    ssq *= tmp2 * tmp2;
                    ssq += one;
                    dev = reps * ::sqrt(ssq);
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
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index size,
        Scalar eps, const T *gamma_, const T *beta_, const T *sumnorm_,
        T *dst_)
    noexcept
//! Renormalize buffer along middle axis
/*! Provided m-by-k-by-n output array dst is renormalized along second axis
 * with k elements. The following operations is applied:
 *      dst[i, :, j] := (dst[i, :, j]-mean(i, j)) / sqrt(var(i, j)+eps)
 *          * gamma + beta
 * where mean and var functions are computed as follows:
 *      mean(i, j) = sumnorm[0, i, j] / size
 *      var(i, j) = sumnorm[1, i, j]^2/size - mean(i,j)^2
 *
 * @param[in] m: Size of the first mode of dst and sumnorm arrays
 * @param[in] n: Size of the last mode of dst and sumnorm arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] size: Number of elements used to calculate sum and Euclidean norm
 * @param[in] eps: Regularization parameter for variance
 * @param[in] gamma_: Deviation for the renormalized output
 * @param[in] beta_: Mean value for the renormalized output
 * @param[in] sumnorm_: Sums and norms of slices
 * @param[in] dst_: Contiguous output array
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    dim3 blocks(16, 16), threads(8, 4);
    using Y = typename CUDAComputeType<T>::value;
    auto gamma = reinterpret_cast<const Y *>(gamma_);
    auto beta = reinterpret_cast<const Y *>(beta_);
    auto sumnorm = reinterpret_cast<const Y *>(sumnorm_);
    auto dst = reinterpret_cast<Y *>(dst_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(m, n, k, size, Y{eps},
            gamma, beta, sumnorm, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index size,
        Scalar eps, const fp32_t *gamma, const fp32_t *beta,
        const fp32_t *sumnorm, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index size,
        Scalar eps, const fp64_t *gamma, const fp64_t *beta,
        const fp64_t *sumnorm, fp64_t *dst)
    noexcept;

} // namespace nntile::kernel::normalize
