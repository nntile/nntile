/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cuda.cu
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include <iostream>

#include "nntile/kernel/maxsumexp/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::maxsumexp
{

/**
 * This implementation is taken from 3c4a8ee08f66732d67789f851c6bff788e41fd38.
 */
// clang-format off
template<typename T>
static __global__
void cuda_kernel(Index m, Index m_per_block, Index n, Index n_per_block,
        Index k, Index mk, const T * __restrict__ src,
        T * __restrict__ maxsumexp)
{
    Index i1_block = blockIdx.y, i2_block = blockIdx.z,
          i0_start = threadIdx.x, i0_step = blockDim.x;

    using Y = typename T::repr_t;
    constexpr Y zero = 0.0, one = 1.0;
    if(i0_start < k)
    {
        for(Index i1 = i1_block*m_per_block;
                i1 < (i1_block+1)*m_per_block and i1 < m; ++i1)
        {
            for(Index i2 = i2_block*n_per_block;
                    i2 < (i2_block+1)*n_per_block and i2 < n; ++i2)
            {
                // Get max and sum of exponents of a corresponding slice
                const T *src_slice = src + i2*mk + i1;
                // Init max and sum
                Y max_val = Y{src_slice[i0_start*m]};
                Y sum_val = one;
                // Cycle over slice of input buffer
                for(Index i0 = i0_start+i0_step; i0 < k; i0 += i0_step)
                {
                    // Read value from source
                    Y val = Y{src_slice[i0*m]};
                    // Ignore -inf value, which comes from mask
                    if(::isinf(val))
                    {
                        continue;
                    }
                    // Update max and sum of exponents
                    if(max_val < val)
                    {
                        sum_val = sum_val*(::exp(max_val-val)) + one;
                        max_val = val;
                    }
                    else
                    {
                        sum_val += ::exp(val-max_val);
                    }
                }
                // Per-block of threads max and sum of exponents
                volatile __shared__ Y block_max_val;
                __shared__ Y block_sum_val;
                // Init shared values in the i0_start==0 thread
                if(i0_start == 0)
                {
                    block_max_val = max_val;
                    block_sum_val = zero;
                }
                // Other threads wait until initialization is done
                __syncthreads();
                // Update max at first
                while(block_max_val < max_val)
                {
                    block_max_val = max_val;
                }
                // Sync with all other threads to get per-block max finally
                __syncthreads();
                // Accumulate per-block sum of finite values
                if(not ::isinf(max_val))
                {
                    sum_val *= ::exp(max_val - block_max_val);
                    atomicAdd(&block_sum_val, sum_val);
                }
                __syncthreads();
                // Update output iff per-block sum is not zero
                if(i0_start == 0 and block_sum_val > 0)
                {
                    // Get per-block max and sum of exponents into local variables
                    max_val = block_max_val;
                    sum_val = block_sum_val;
                    Index dst_offset = i1 + i2*m;
                    // Now max_val is finite, we need to accumulate sum of exponents
                    // with the data in global memory
                    Y max_output;
                    Y sum_output = Y{maxsumexp[2*dst_offset+1]};
                    // If data was not yet initialised, just overwrite it
                    if(sum_output == zero)
                    {
                        max_output = max_val;
                        sum_output = sum_val;
                    }
                    // Accumulate otherwise
                    else
                    {
                        max_output = Y{maxsumexp[2*dst_offset]};
                        if(max_val < max_output)
                        {
                            sum_val *= ::exp(max_val - max_output);
                        }
                        else
                        {
                            sum_output *= ::exp(max_output - max_val);
                            max_output = max_val;
                        }
                        sum_output += sum_val;
                    }
                    maxsumexp[2*dst_offset] = T{max_output};
                    maxsumexp[2*dst_offset+1] = T{sum_output};
                }
            }
        }
    }
}

template <typename T>
void LaunchMaxSumExp1(cudaStream_t stream, Index m, Index n, Index k,
                      const T *src_, T *maxsumexp_)
    noexcept
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(32, 1, 1), blocks(1, m, n);
    Index m_per_block = 1, n_per_block = 1;
    if(m > 65535)
    {
        m_per_block = (m+65534) / 65535;
        blocks.y = (m+m_per_block-1) / m_per_block;
    }
    if(n > 65535)
    {
        n_per_block = (n+65534) / 65535;
        blocks.z = (n+n_per_block-1) / n_per_block;
    }
    // using Y = typename CUDAComputeType<T>::value;
    // auto src = reinterpret_cast<const Y *>(src_);
    // auto maxsumexp = reinterpret_cast<Y *>(maxsumexp_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, m_per_block, n,
            n_per_block, k, m*k, src_, maxsumexp_);
}
// clang-format on

template void LaunchMaxSumExp1<fp32_t>(cudaStream_t stream, Index m, Index n,
                                       Index k, const fp32_t *src,
                                       fp32_t *maxsumexp) noexcept;

template void LaunchMaxSumExp1<fp64_t>(cudaStream_t stream, Index m, Index n,
                                       Index k, const fp64_t *src,
                                       fp64_t *maxsumexp) noexcept;

template void LaunchMaxSumExp1<bf16_t>(cudaStream_t stream, Index m, Index n,
                                       Index k, const bf16_t *src,
                                       bf16_t *maxsumexp) noexcept;

//extern __shared__ float extent[]; // User-managed cache on device.
//
//size_t constexpr kMaxBlockSize = 512;
//
//template <typename T, uint32_t kBlockSize>
//__device__ void BlockMaxReduce(volatile T *acc, uint32_t tid) {
//    if constexpr (kBlockSize >= 1024) {
//        if (tid < 512) {
//            acc[tid] = ::fmax(acc[tid], acc[tid + 512]);
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 512) {
//        if (tid < 256) {
//            acc[tid] = ::fmax(acc[tid], acc[tid + 256]);
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 256) {
//        if (tid < 128) {
//            acc[tid] = ::fmax(acc[tid], acc[tid + 128]);
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 128) {
//        if (tid < 64) {
//            acc[tid] = ::fmax(acc[tid], acc[tid + 64]);
//        }
//        __syncthreads();
//    }
//}
//
//template <typename T, uint32_t kBlockSize, uint32_t kStride>
//__device__ void WarpMaxReduceRound(volatile T *acc, uint32_t tid) {
//    if constexpr (kBlockSize >= 2 * kStride) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + kStride]);
//    }
//}
//
//template <typename T, uint32_t kBlockSize>
//__device__ void WarpMaxReduce(volatile T *acc, uint32_t tid) {
//    if constexpr (kBlockSize >= 64) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 32]);
//    }
//    if constexpr (kBlockSize >= 32) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 16]);
//    }
//    if constexpr (kBlockSize >= 16) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 8]);
//    }
//    if constexpr (kBlockSize >= 8) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 4]);
//    }
//    if constexpr (kBlockSize >= 4) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 2]);
//    }
//    if constexpr (kBlockSize >= 2) {
//        acc[tid] = ::fmax(acc[tid], acc[tid + 1]);
//    }
//}
//
//template <typename T, uint32_t kBlockSize>
//__device__ void BlockSumExpReduce(volatile T *acc, uint32_t tid) {
//    if constexpr (kBlockSize >= 1024) {
//        if (tid < 512) {
//            acc[tid] = acc[tid] + acc[tid + 512];
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 512) {
//        if (tid < 256) {
//            acc[tid] = acc[tid] + acc[tid + 256];
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 256) {
//        if (tid < 128) {
//            acc[tid] = acc[tid] + acc[tid + 128];
//        }
//        __syncthreads();
//    }
//    if constexpr (kBlockSize >= 128) {
//        if (tid < 64) {
//            acc[tid] = acc[tid] + acc[tid + 64];
//        }
//        __syncthreads();
//    }
//}
//
//template <typename T, uint32_t kBlockSize>
//__device__ void WarpSumExpReduce(volatile T *acc, uint32_t tid) {
//    if constexpr (kBlockSize >= 64) {
//        acc[tid] = acc[tid] + acc[tid + 32];
//    }
//    if constexpr (kBlockSize >= 32) {
//        acc[tid] = acc[tid] + acc[tid + 16];
//    }
//    if constexpr (kBlockSize >= 16) {
//        acc[tid] = acc[tid] + acc[tid + 8];
//    }
//    if constexpr (kBlockSize >= 8) {
//        acc[tid] = acc[tid] + acc[tid + 4];
//    }
//    if constexpr (kBlockSize >= 4) {
//        acc[tid] = acc[tid] + acc[tid + 2];
//    }
//    if constexpr (kBlockSize >= 2) {
//        acc[tid] = acc[tid] + acc[tid + 1];
//    }
//}
//
//template <typename T, uint32_t kBlockSize>
//__global__ void MaxSumExp3(Index m, Index n, Index k, Index mk,
//                           T const *__restrict__ src, T *__restrict__ dst) {
//    // Memory model of user-maneged cache in shared memory.
//    size_t const data_size = blockDim.x * blockDim.y * blockDim.z;
//    T *cache = reinterpret_cast<T *>(extent); // Mirror of global memory.
//    // Accumulator for max-reduction and sum-reduction.
//    T *acc = reinterpret_cast<T *>(cache) + data_size;
//
//    // Obtain global and local position of the current thread.
//    auto tid = threadIdx.y;
//    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
//    auto jx = threadIdx.y + blockDim.y * blockIdx.y;
//    auto kx = threadIdx.z + blockDim.z * blockIdx.z;
//    bool out_of_scope = ix >= m || jx >= k || kx >= n;
//
//    // auto it = (2 * kBlockSize) * blockIdx.y + tid;
//    // auto grid_size = (2 * kBlockSize) * gridDim.y;
//    // auto data = src + (ix + mk * kx);
//
//    // Load data from global memory to user-managed cache in shared memory.
//    if (out_of_scope) {
//        cache[tid] = -INFINITY;
//        acc[tid] = -INFINITY;
//    } else {
//        cache[tid] = src[ix + m * jx + mk * kx];
//        acc[tid] = cache[tid];
//    }
//    __syncthreads();
//
//    // Per-block max-reduction in shared memory.
//    BlockMaxReduce<T, kBlockSize>(acc, tid);
//    if (tid < 32) {
//        WarpMaxReduce<T, kBlockSize>(acc, tid);
//    }
//
//    // Per-block sumexp-reduction in shared memory.
//    T const max = acc[0];
//    acc[tid] = exp(cache[tid] - max);
//    __syncthreads();
//
//    BlockSumExpReduce<T, kBlockSize>(acc, tid);
//    if (tid < 32) {
//        WarpSumExpReduce<T, kBlockSize>(acc, tid);
//    }
//
//    // Store in global memory (output buffer) in theads from X-Z plane.
//    if (tid == 0) {
//        // Contingues tuple of (max, sum). Update accumulants in-place.
//        auto out = dst + 2 * (ix + m * kx);
//        if (auto diff = max - out[0]; diff > 0) {
//            out[0] = max;
//            out[1] = out[1] * exp(-diff) + acc[tid];
//        } else {
//            out[1] = out[1] + exp(diff) * acc[tid];
//        }
//    }
//}
//
//template <typename T> constexpr T ceil2(T value) {
//    static_assert(std::is_integral<T>::value, "integral type expected");
//    value--;
//    // Divide by 2^k for consecutive doublings of k up to 256,
//    // and then or the results.
//    value |= value >> 1;
//    value |= value >> 2;
//    value |= value >> 4;
//    if constexpr (sizeof(value) >= 2) {
//        value |= value >> 8;
//    }
//    if constexpr (sizeof(value) >= 4) {
//        value |= value >> 16;
//    }
//    if constexpr (sizeof(value) >= 8) {
//        value |= value >> 32;
//    }
//    if constexpr (sizeof(value) >= 16) {
//        value |= value >> 64;
//    }
//    if constexpr (sizeof(value) >= 32) {
//        value |= value >> 128;
//    }
//    // The result is a number of 1 bits equal to the number
//    // of bits in the original number, plus 1. That's the
//    // next highest power of 2.
//    return ++value;
//}
//
//template <typename T>
//void LaunchMaxSumExp3(cudaStream_t stream, Index m, Index n, Index k,
//                      T const *src, T *dst) noexcept {
//    size_t block_size = ceil2(k);
//    if (block_size > kMaxBlockSize) {
//        block_size = kMaxBlockSize;
//    }
//
//    dim3 threads(1, block_size, 1);
//    auto noblocks = (k - 1) / threads.y + 1;
//    dim3 blocks(m, noblocks, n);
//    size_t smem = 2 * threads.x * threads.y * threads.z * sizeof(T);
//
//    if (blocks.y > 1) {
//        std::cerr << "unsupported thread block size" << std::endl;
//        std::terminate();
//    }
//
//    switch (threads.y) {
//    case 1024:
//        MaxSumExp3<T, 1024>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 512:
//        MaxSumExp3<T, 512>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 256:
//        MaxSumExp3<T, 256>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 128:
//        MaxSumExp3<T, 128>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 64:
//        MaxSumExp3<T, 64>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 32:
//        MaxSumExp3<T, 32>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 16:
//        MaxSumExp3<T, 16>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 8:
//        MaxSumExp3<T, 8>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 4:
//        MaxSumExp3<T, 4>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 2:
//        MaxSumExp3<T, 2>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    case 1:
//        MaxSumExp3<T, 1>
//            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
//        break;
//    default:
//        std::cerr << "unsupported thread block size" << std::endl;
//        break;
//    }
//}
//
//template void LaunchMaxSumExp3<fp32_t>(cudaStream_t stream, Index m, Index n,
//                                       Index k, const fp32_t *src,
//                                       fp32_t *maxsumexp) noexcept;
//
//template void LaunchMaxSumExp3<fp64_t>(cudaStream_t stream, Index m, Index n,
//                                       Index k, const fp64_t *src,
//                                       fp64_t *maxsumexp) noexcept;

template <typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
          T *maxsumexp)
    noexcept
{
    LaunchMaxSumExp1(stream, m, n, k, src, maxsumexp);
}

template void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *maxsumexp) noexcept;

template void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *maxsumexp) noexcept;

template void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *src, bf16_t *maxsumexp) noexcept;

template void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_fast_tf32_t *src, fp32_fast_tf32_t *maxsumexp) noexcept;

} // namespace nntile::kernel::maxsumexp
