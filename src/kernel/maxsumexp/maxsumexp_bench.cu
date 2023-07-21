#include <cstddef>
#include <iterator>
#include <stdexcept>

#include <cuda.h>
#include <nvbench/launch.cuh>
#include <nvbench/nvbench.cuh>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "nntile/base_types.hh"
#include "nntile/kernel/maxsumexp.hh"
#include "nntile/kernel/maxsumexp/cuda.hh"

using nntile::Index;

namespace maxsumexp = nntile::kernel::maxsumexp;

enum class Device : int {
    kCPU = 0,
    kCUDA = 1,
};

template <typename T, Device device> struct Array;

template <typename T> struct Array<T, Device::kCUDA> {
    size_t size;
    T *data = nullptr;
    cudaError_t status = cudaSuccess;

    Array(size_t size) noexcept : size{size} {
        status = cudaMalloc(&data, size * sizeof(T));
    }

    ~Array(void) {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    operator bool(void) const {
        return status == cudaError_t::cudaSuccess;
    }

    template <typename U> U *as(void) {
        return reinterpret_cast<U *>(data);
    }

    template <typename U> U *as(void) const {
        return reinterpret_cast<U const *>(data);
    }
};

template <typename T>
__global__ void Copy(Index n, Index m, Index k, T const *src, T *dst) {
    auto ix = threadIdx.y + blockIdx.y * blockDim.y;
    auto jx = threadIdx.x + blockIdx.x * blockDim.x;
    auto kx = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= n || kx >= m) {
        return;
    }
    for (auto jt = 0; jt != k; ++jt) {
        auto value = src[ix + n * (jx + jt) + n * m * kx];
        dst[ix + n * kx + 0] = value;
        dst[ix + n * kx + 1] = value;
    }
}

void BenchCopy(nvbench::state &state) {
    auto batch_size = static_cast<int>(state.get_int64("batch_size"));
    auto seq_len = static_cast<int>(state.get_int64("seq_len"));
    auto src = Array<float, Device::kCUDA>(batch_size * seq_len * seq_len);
    auto dst = Array<float, Device::kCUDA>(batch_size * seq_len * 2);

    // Request throughput stats.
    state.add_element_count(src.size);
    state.add_global_memory_reads<float>(src.size);
    state.add_global_memory_writes<float>(dst.size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        dim3 threads(256);
        dim3 blocks(1, batch_size, seq_len);
        Copy<float><<<threads, blocks, 0, launch.get_stream()>>>(
            batch_size, seq_len, seq_len, src.as<float>(), dst.as<float>());
        cudaStreamSynchronize(launch.get_stream());
    });
}

NVBENCH_BENCH(BenchCopy)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256});

void BenchMaxSumExp(nvbench::state &state) {
    auto batch_size = static_cast<int>(state.get_int64("batch_size"));
    auto seq_len = static_cast<int>(state.get_int64("seq_len"));
    auto src = Array<float, Device::kCUDA>(batch_size * seq_len * seq_len);
    auto dst = Array<float, Device::kCUDA>(batch_size * seq_len * 2);

    // Request throughput stats.
    state.add_element_count(src.size);
    state.add_global_memory_reads<float>(src.size);
    state.add_global_memory_writes<float>(dst.size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        maxsumexp::cuda(launch.get_stream(), batch_size, seq_len, seq_len,
                        src.as<float>(), dst.as<float>());
        cudaStreamSynchronize(launch.get_stream());
    });
}

NVBENCH_BENCH(BenchMaxSumExp)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256, 1024});

template <typename T, typename Distance = std::intptr_t, typename Pointer = T *,
          typename Reference = T &>
class StridedIterator : public std::iterator<std::random_access_iterator_tag,
                                             typename std::remove_cv<T>::type,
                                             Distance, Pointer, Reference> {
public:
    // TODO(@bershatsky): Use iterator traits.
    using difference_type = Distance;
    using pointer = Pointer;
    using reference = Reference;

public:
    T *ptr_;
    difference_type stride_;

public:
    __device__ StridedIterator(void) = delete;

    __device__ StridedIterator(StridedIterator const &that) noexcept
        : ptr_{that.ptr_}, stride_{that.stride_} {
    }

    __device__
    StridedIterator(T *ptr, difference_type stride = difference_type()) noexcept
        : ptr_{ptr}, stride_{stride} {
    }

    __device__ StridedIterator &operator++(void) {
        ptr_ += stride_;
        return *this;
    }

    __device__ StridedIterator operator++(int) {
        StridedIterator it(*this);
        ++(*this);
        return it;
    }

    __device__ StridedIterator &operator+=(difference_type offset) {
        ptr_ += offset * stride_;
        return *this;
    }

    __device__ StridedIterator operator+(difference_type offset) const {
        auto ptr = ptr_ + offset * stride_;
        return {ptr, stride_};
    }

    __device__ difference_type operator-(StridedIterator const &that) const {
        auto offset = ptr_ - that.ptr_;
        return offset / stride_;
    }

    __device__ bool operator==(StridedIterator const &that) const {
        return ptr_ == that.ptr_;
    }

    __device__ bool operator!=(StridedIterator const &that) const {
        return !(*this == that);
    }

    __device__ reference operator*(void) const {
        return *ptr_;
    }

    __device__ reference &operator*(void) {
        return *ptr_;
    }

    __device__ reference operator[](size_t index) const {
        return *(ptr_ + stride_ * index);
    }
};

template <typename T>
__global__ void MaxSumExp2(Index m, Index n, Index k, Index mk,
                           T const *__restrict__ src, T *__restrict__ dst) {
    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
    auto kx = threadIdx.z + blockDim.z * blockIdx.z;
    if (ix >= m || kx >= n) {
        return;
    }

    StridedIterator begin(src + ix + mk * kx, m);
    StridedIterator end(src + ix + mk * (kx + 1), m);

    auto max = thrust::max_element(thrust::device, begin, end);
    auto unary = [x_max = *max](auto x) { return std::exp(x - x_max); };
    auto sum = thrust::transform_reduce(thrust::device, begin, end, unary, T(),
                                        thrust::plus<T>());

    dst = dst + ix + mk * kx;
    dst[0] = *max;
    dst[1] = sum;
}

using T = float;
void LaunchMaxSumExp2(cudaStream_t stream, Index m, Index n, Index k,
                      T const *src, T *dst) noexcept {
    dim3 threads(m, 1, n);
    dim3 blocks(1);
    MaxSumExp2<T><<<blocks, threads, 0, stream>>>(m, n, k, m * k, src, dst);
}

void BenchMaxSumExp2(nvbench::state &state) {
    auto batch_size = static_cast<int>(state.get_int64("batch_size"));
    auto seq_len = static_cast<int>(state.get_int64("seq_len"));
    auto src = Array<float, Device::kCUDA>(batch_size * seq_len * seq_len);
    auto dst = Array<float, Device::kCUDA>(batch_size * seq_len * 2);

    // Request throughput stats.
    state.add_element_count(src.size);
    state.add_global_memory_reads<float>(src.size);
    state.add_global_memory_writes<float>(dst.size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        LaunchMaxSumExp2(launch.get_stream(), batch_size, seq_len, seq_len,
                         src.as<float>(), dst.as<float>());
        cudaStreamSynchronize(launch.get_stream());
    });
}

NVBENCH_BENCH(BenchMaxSumExp2)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256, 1024});

extern __shared__ float extent[]; // User-managed cache on device.

size_t constexpr kMaxBlockSize = 512;

template <typename T, uint32_t kBlockSize>
__device__ void BlockMaxReduce(volatile T *acc, uint32_t tid) {
    if constexpr (kBlockSize >= 1024) {
        if (tid < 512) {
            acc[tid] = std::max(acc[tid], acc[tid + 512]);
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 512) {
        if (tid < 256) {
            acc[tid] = std::max(acc[tid], acc[tid + 256]);
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 256) {
        if (tid < 128) {
            acc[tid] = std::max(acc[tid], acc[tid + 128]);
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 128) {
        if (tid < 64) {
            acc[tid] = std::max(acc[tid], acc[tid + 64]);
        }
        __syncthreads();
    }
}

template <typename T, uint32_t kBlockSize, uint32_t kStride>
__device__ void WarpMaxReduceRound(volatile T *acc, uint32_t tid) {
    if constexpr (kBlockSize >= 2 * kStride) {
        acc[tid] = std::max(acc[tid], acc[tid + kStride]);
    }
}

template <typename T, uint32_t kBlockSize>
__device__ void WarpMaxReduce(volatile T *acc, uint32_t tid) {
    if constexpr (kBlockSize >= 64) {
        acc[tid] = std::max(acc[tid], acc[tid + 32]);
    }
    if constexpr (kBlockSize >= 32) {
        acc[tid] = std::max(acc[tid], acc[tid + 16]);
    }
    if constexpr (kBlockSize >= 16) {
        acc[tid] = std::max(acc[tid], acc[tid + 8]);
    }
    if constexpr (kBlockSize >= 8) {
        acc[tid] = std::max(acc[tid], acc[tid + 4]);
    }
    if constexpr (kBlockSize >= 4) {
        acc[tid] = std::max(acc[tid], acc[tid + 2]);
    }
    if constexpr (kBlockSize >= 2) {
        acc[tid] = std::max(acc[tid], acc[tid + 1]);
    }
}

template <typename T, uint32_t kBlockSize>
__device__ void BlockSumExpReduce(volatile T *acc, uint32_t tid) {
    if constexpr (kBlockSize >= 1024) {
        if (tid < 512) {
            acc[tid] = acc[tid] + acc[tid + 512];
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 512) {
        if (tid < 256) {
            acc[tid] = acc[tid] + acc[tid + 256];
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 256) {
        if (tid < 128) {
            acc[tid] = acc[tid] + acc[tid + 128];
        }
        __syncthreads();
    }
    if constexpr (kBlockSize >= 128) {
        if (tid < 64) {
            acc[tid] = acc[tid] + acc[tid + 64];
        }
        __syncthreads();
    }
}

template <typename T, uint32_t kBlockSize>
__device__ void WarpSumExpReduce(volatile T *acc, uint32_t tid) {
    if constexpr (kBlockSize >= 64) {
        acc[tid] = acc[tid] + acc[tid + 32];
    }
    if constexpr (kBlockSize >= 32) {
        acc[tid] = acc[tid] + acc[tid + 16];
    }
    if constexpr (kBlockSize >= 16) {
        acc[tid] = acc[tid] + acc[tid + 8];
    }
    if constexpr (kBlockSize >= 8) {
        acc[tid] = acc[tid] + acc[tid + 4];
    }
    if constexpr (kBlockSize >= 4) {
        acc[tid] = acc[tid] + acc[tid + 2];
    }
    if constexpr (kBlockSize >= 2) {
        acc[tid] = acc[tid] + acc[tid + 1];
    }
}

template <typename T, uint32_t kBlockSize>
__global__ void MaxSumExp3(Index m, Index n, Index k, Index mk,
                           T const *__restrict__ src, T *__restrict__ dst) {
    // Memory model of user-maneged cache in shared memory.
    size_t const data_size = blockDim.x * blockDim.y * blockDim.z;
    T *cache = reinterpret_cast<T *>(extent); // Mirror of global memory.
    // Accumulator for max-reduction and sum-reduction.
    T *acc = reinterpret_cast<T *>(cache) + data_size;

    // Obtain global and local position of the current thread.
    auto tid = threadIdx.y;
    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
    auto jx = threadIdx.y + blockDim.y * blockIdx.y;
    auto kx = threadIdx.z + blockDim.z * blockIdx.z;
    bool out_of_scope = ix >= m || jx >= k || kx >= n;

    // auto it = (2 * kBlockSize) * blockIdx.y + tid;
    // auto grid_size = (2 * kBlockSize) * gridDim.y;
    // auto data = src + (ix + mk * kx);

    // Load data from global memory to user-managed cache in shared memory.
    if (out_of_scope) {
        cache[tid] = -std::numeric_limits<T>::infinity();
        acc[tid] = -std::numeric_limits<T>::infinity();
    } else {
        cache[tid] = src[ix + m * jx + mk * kx];
        acc[tid] = cache[tid];
    }
    __syncthreads();

    // Per-block max-reduction in shared memory.
    BlockMaxReduce<T, kBlockSize>(acc, tid);
    if (tid < 32) {
        WarpMaxReduce<T, kBlockSize>(acc, tid);
    }

    // Per-block sumexp-reduction in shared memory.
    T const max = acc[0];
    acc[tid] = std::exp(cache[tid] - max);
    __syncthreads();

    BlockSumExpReduce<T, kBlockSize>(acc, tid);
    if (tid < 32) {
        WarpSumExpReduce<T, kBlockSize>(acc, tid);
    }

    // Store in global memory (output buffer) in theads from X-Z plane.
    if (tid == 0) {
        // Contingues tuple of (max, sum). Update accumulants in-place.
        auto out = dst + 2 * (ix + m * kx);
        if (auto diff = max - out[0]; diff > 0) {
            out[0] = max;
            out[1] = out[1] * std::exp(-diff) + acc[tid];
        } else {
            out[1] = out[1] + std::exp(diff) * acc[tid];
        }
    }
}

template <typename T> constexpr T ceil2(T value) {
    static_assert(std::is_integral<T>::value, "integral type expected");
    value--;
    // Divide by 2^k for consecutive doublings of k up to 256,
    // and then or the results.
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    if constexpr (sizeof(value) >= 2) {
        value |= value >> 8;
    }
    if constexpr (sizeof(value) >= 4) {
        value |= value >> 16;
    }
    if constexpr (sizeof(value) >= 8) {
        value |= value >> 32;
    }
    if constexpr (sizeof(value) >= 16) {
        value |= value >> 64;
    }
    if constexpr (sizeof(value) >= 32) {
        value |= value >> 128;
    }
    // The result is a number of 1 bits equal to the number
    // of bits in the original number, plus 1. That's the
    // next highest power of 2.
    return ++value;
}

template <typename T>
void LaunchMaxSumExp3(cudaStream_t stream, Index m, Index n, Index k,
                      T const *src, T *dst) noexcept {
    size_t block_size = ceil2(k);
    if (block_size > kMaxBlockSize) {
        block_size = kMaxBlockSize;
    }

    dim3 threads(1, block_size, 1);
    auto noblocks = (k - 1) / threads.y + 1;
    dim3 blocks(m, noblocks, n);
    size_t smem = 2 * threads.x * threads.y * threads.z * sizeof(T);

    if (blocks.y > 1) {
        std::cerr << "unsupported thread block size" << std::endl;
        std::terminate();
    }

    switch (threads.y) {
    case 1024:
        MaxSumExp3<T, 1024>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 512:
        MaxSumExp3<T, 512>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 256:
        MaxSumExp3<T, 256>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 64:
        MaxSumExp3<T, 64>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 32:
        MaxSumExp3<T, 32>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 16:
        MaxSumExp3<T, 16>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 8:
        MaxSumExp3<T, 8>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 4:
        MaxSumExp3<T, 4>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 2:
        MaxSumExp3<T, 2>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    case 1:
        MaxSumExp3<T, 1>
            <<<blocks, threads, smem, stream>>>(m, n, k, m * k, src, dst);
        break;
    default:
        std::cerr << "unsupported thread block size" << std::endl;
        break;
    }
}

void BenchMaxSumExp3(nvbench::state &state) {
    auto batch_size = static_cast<int>(state.get_int64("batch_size"));
    auto seq_len = static_cast<int>(state.get_int64("seq_len"));
    auto src = Array<float, Device::kCUDA>(batch_size * seq_len * seq_len);
    auto dst = Array<float, Device::kCUDA>(batch_size * seq_len * 2);

    // Request throughput stats.
    state.add_element_count(src.size);
    state.add_global_memory_reads<float>(src.size);
    state.add_global_memory_writes<float>(dst.size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
        LaunchMaxSumExp3(launch.get_stream(), batch_size, seq_len, seq_len,
                         src.as<float>(), dst.as<float>());
        // cudaStreamSynchronize(launch.get_stream());
        cudaDeviceSynchronize();
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::cout << status << " " << cudaGetErrorName(status) << " "
                      << cudaGetErrorString(status) << std::endl;
            throw std::runtime_error("CUDA error");
        }
    });
}

NVBENCH_BENCH(BenchMaxSumExp3)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256, 1024});
