/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/maxsumexp.cc
 * Max and sums of exponents of a buffer on CPU
 *
 * @version 1.1.0
 * */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <strings.h>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "nntile/kernel/maxsumexp.hh"

using namespace nntile;
using nntile::kernel::maxsumexp::cpu;
using nntile::kernel::maxsumexp::LaunchMaxSumExp1;
using nntile::kernel::maxsumexp::LaunchMaxSumExp3;

// TileShape (aka batch size, reduced size, seq len) represents a shape of tile
// which is reduced along the middle (seq len) axis.
using TileShape = std::tuple<int, int, int>;

class MaxSumExpCPU : public ::testing::TestWithParam<TileShape> {
protected:
    void SetUp(void) override {
        batch_size_ = std::get<0>(GetParam());
        seq_len_ = std::get<1>(GetParam());
        reduced_size_ = std::get<2>(GetParam());
    }

    template <typename T> std::vector<T> GenerateData(void) {
        Index m = batch_size_, n = seq_len_, k = reduced_size_;
        std::vector<T> src(m * n * k);
        for (Index i0 = 0; i0 < m; ++i0) {
            for (Index i1 = 0; i1 < n; ++i1) {
                for (Index i2 = 0; i2 < k; ++i2) {
                    src[(i1 * k + i2) * m + i0] = T(i0 + i1 + i2) / T{10};
                }
            }
        }
        return src;
    }

    template <typename T> void AssertSimple(std::vector<T> const &maxsumexp) {
        T constexpr eps = std::numeric_limits<T>::epsilon();
        Index m = batch_size_, n = seq_len_, k = reduced_size_;
        for (Index i0 = 0; i0 < m; ++i0) {
            for (Index i1 = 0; i1 < n; ++i1) {
                Index a = i0 + i1;
                T max_ref = T(a + k - 1) / T{10};
                T max = maxsumexp[2 * (i1 * m + i0)];
                ASSERT_EQ(max, max_ref) << max << " vs " << max_ref << " at ("
                                        << i0 << ", " << i1 << ")";
                T sum_ref = 0;
                for (Index i2 = 0; i2 < k; ++i2) {
                    sum_ref += std::exp(T(i2 - k + 1) / T{10});
                }
                T sum = maxsumexp[2 * (i1 * m + i0) + 1];
                auto rerr = std::abs(sum / sum_ref - T{1});
                ASSERT_TRUE(rerr <= 10 * eps)
                    << sum << " vs " << sum_ref << " at (" << i0 << ", " << i1
                    << ")";
            }
        }
    }

    template <typename T>
    void AssertDetailed(std::vector<T> const &maxsumexp,
                        std::vector<T> const &maxsumexp_copy) {
        T constexpr eps = std::numeric_limits<T>::epsilon();
        Index m = batch_size_, n = seq_len_, k = reduced_size_;
        for (Index i0 = 0; i0 < m; ++i0) {
            for (Index i1 = 0; i1 < n; ++i1) {
                Index i = 2 * (i1 * m + i0);
                ASSERT_EQ(maxsumexp[i], maxsumexp_copy[i]);
                auto rerr =
                    std::abs(maxsumexp[i + 1] / maxsumexp_copy[i + 1] - T{2});
                ASSERT_TRUE(rerr <= 10 * eps)
                    << maxsumexp[i + 1] << " vs " << maxsumexp_copy[i + 1]
                    << " at (" << i0 << ", " << i1 << ")";
            }
        }
    }

    template <typename T> void RunTest(void) {
        Index m = batch_size_, n = seq_len_, k = reduced_size_;

        std::vector<T> src = GenerateData<T>();
        std::vector<T> maxsumexp(2 * batch_size_ * seq_len_);
        cpu<T>(m, n, k, &src[0], &maxsumexp[0]);
        AssertSimple<T>(maxsumexp);

        std::vector<T> maxsumexp_copy(maxsumexp);
        cpu<T>(m, n, k, &src[0], &maxsumexp[0]);
        AssertDetailed<T>(maxsumexp, maxsumexp_copy);
    }

protected:
    int batch_size_;
    int reduced_size_;
    int seq_len_;
};

TEST_P(MaxSumExpCPU, FP64) {
    RunTest<double>();
}

TEST_P(MaxSumExpCPU, FP32) {
    RunTest<float>();
}

INSTANTIATE_TEST_SUITE_P(Kernel, MaxSumExpCPU,
                         ::testing::Values(std::make_tuple(1, 9, 10),
                                           std::make_tuple(8, 9, 1),
                                           std::make_tuple(8, 1, 10),
                                           std::make_tuple(4, 7, 8),
                                           std::make_tuple(32, 1024, 1024)));

template <typename T> void FreeDeviceArray(T *ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
};

template <typename T>
using device_ptr = std::unique_ptr<T, decltype(&FreeDeviceArray<T>)>;

template <typename T>
device_ptr<T> MakeDevicePtrLike(std::vector<T> const &src) {
    T *ptr = nullptr;
    cudaMalloc(&ptr, src.size() * sizeof(T));
    return device_ptr<T>(ptr, FreeDeviceArray);
}

class MaxSumExpCUDA : public MaxSumExpCPU {
protected:
    cudaStream_t stream_;

protected:
    void SetUp(void) override {
        MaxSumExpCPU::SetUp();
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            // TODO(@bershatsky): Handle CUDA errors.
        }
    }

    void TearDown(void) override {
        cudaStreamDestroy(stream_);
        MaxSumExpCPU::TearDown();
    }

    template <typename T>
    void LaunchKernel(Index m, Index n, Index k, T const *input, T *output,
                      std::vector<T> &result) {
        LaunchKernel<T>(stream_, m, n, k, input, output);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        cudaMemcpy(result.data(), output, result.size() * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

    template <typename T> void RunTest(void) {
        Index m = batch_size_, n = seq_len_, k = reduced_size_;

        std::vector<T> src = GenerateData<T>();
        std::vector<T> maxsumexp(2 * batch_size_ * seq_len_);
        auto source_ = MakeDevicePtrLike(src);
        auto target_ = MakeDevicePtrLike(maxsumexp);
        cudaMemcpy(source_.get(), src.data(), src.size() * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(target_.get(), maxsumexp.data(),
                   maxsumexp.size() * sizeof(T), cudaMemcpyHostToDevice);
        LaunchKernel(m, n, k, source_.get(), target_.get(), maxsumexp);
        AssertSimple<T>(maxsumexp);

        std::vector<T> maxsumexp_copy(maxsumexp);
        LaunchKernel(m, n, k, source_.get(), target_.get(), maxsumexp);
        AssertDetailed<T>(maxsumexp, maxsumexp_copy);
    }

protected:
    template <typename T>
    void LaunchKernel(cudaStream_t stream_, Index m, Index n, Index k,
                      T const *input, T *output) {
        LaunchMaxSumExp1<T>(stream_, m, n, k, input, output);
    }
};

TEST_P(MaxSumExpCUDA, FP64) {
    RunTest<double>();
}

TEST_P(MaxSumExpCUDA, FP32) {
    RunTest<float>();
}

// Test parameters for CUDA tile operations.
static auto const kTestParams =
    ::testing::Values(std::make_tuple(1, 9, 10), std::make_tuple(8, 9, 1),
                      std::make_tuple(8, 1, 10), std::make_tuple(4, 7, 8),
                      std::make_tuple(32, 1024, 1024));

INSTANTIATE_TEST_SUITE_P(Kernel, MaxSumExpCUDA, kTestParams);

class MaxSumExp3CUDA : public MaxSumExpCPU {
protected:
    template <typename T>
    void LaunchKernel(cudaStream_t stream_, Index m, Index n, Index k,
                      T const *input, T *output) {
        LaunchMaxSumExp3<T>(stream_, m, n, k, input, output);
    }
};

TEST_P(MaxSumExp3CUDA, FP64) {
    RunTest<double>();
}

TEST_P(MaxSumExp3CUDA, FP32) {
    RunTest<float>();
}

INSTANTIATE_TEST_SUITE_P(Kernel, MaxSumExp3CUDA, kTestParams);
