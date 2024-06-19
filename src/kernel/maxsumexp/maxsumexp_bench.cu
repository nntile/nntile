/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 */

#include <cstddef>
#include <iterator>
#include <stdexcept>

#include <cuda.h>
#include <nvbench/launch.cuh>
#include <nvbench/nvbench.cuh>

#include "nntile/base_types.hh"
#include "nntile/kernel/maxsumexp.hh"
#include "nntile/kernel/maxsumexp/cuda.hh"

using nntile::Index;
using nntile::kernel::maxsumexp::LaunchMaxSumExp1;
using nntile::kernel::maxsumexp::LaunchMaxSumExp3;

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
        LaunchMaxSumExp1(launch.get_stream(), batch_size, seq_len, seq_len,
                         src.as<float>(), dst.as<float>());
        cudaStreamSynchronize(launch.get_stream());
    });
}

NVBENCH_BENCH(BenchMaxSumExp)
    .add_int64_axis("batch_size", {2, 8, 32})
    .add_int64_axis("seq_len", {64, 256, 1024});

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
