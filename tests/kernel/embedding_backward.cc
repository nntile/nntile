/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/embedding_backward.cc
 * Backward embedding lookup operation for buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/embedding_backward.hh"

// Standard libraries
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cmath>
#include <random>
#include <string>

// Third-party libraries
#include <catch2/catch_all.hpp>

// Other NNTile headers
// CUDA_CHECK definition
#include <nntile/kernel/cuda.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::embedding_backward;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index m; // Size of the first mode of index and embed tensors
    Index n; // Size of the last mode of index and embed tensors
    Index k; // Size of the middle mode of embed tensor
    Index k_start; // Offset of the middle mode of embed tensor
    Index k_size; // Size of the first mode of vocab tensor
    Index vocab_size; // Vocabulary size

    Y eps_check;

    std::vector<int64_t> index;
    std::vector<T> embed;
    std::vector<T> vocab_init;
    std::vector<T> vocab;
    std::vector<T> vocab_ref;
};

// Reference implementation of the embedding_backward operation
template<typename T>
void reference_embedding_backward(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k_size == 0)
    {
        return;
    }

    // Initialize vocab_ref with zeros
    std::fill(data.vocab_ref.begin(), data.vocab_ref.end(), static_cast<T>(0.0));

    for(Index i2 = 0; i2 < data.n; ++i2)
    {
        for(Index i1 = 0; i1 < data.m; ++i1)
        {
            int64_t idx = data.index[i2 * data.m + i1];
            // Output slice of vocabulary
            T *vocab_slice = &data.vocab_ref[data.k_size * idx];
            // Input slice of embedding
            const T *embed_slice = &data.embed[(i2 * data.k + data.k_start) * data.m + i1];

            for(Index i0 = 0; i0 < data.k_size; ++i0)
            {
                vocab_slice[i0] = static_cast<T>(static_cast<Y>(vocab_slice[i0]) + static_cast<Y>(embed_slice[i0 * data.m]));
            }
        }
    }
}

// Enum for data generation strategies
enum class DataGen
{
    PRESET,
    RANDOM
};

// Generates data with preset, deterministic values
template<typename T>
void generate_data(TestData<T>& data, DataGen strategy)
{
    using Y = typename T::repr_t;

    // Initialize indices
    data.index.resize(data.m * data.n);
    switch(strategy)
    {
        case DataGen::PRESET:
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                data.index[i] = (i * 7 + 3) % data.vocab_size; // Ensure indices are within vocab_size
            }
            break;
        case DataGen::RANDOM:
            std::mt19937 gen_idx(42);
            std::uniform_int_distribution<int64_t> dist_idx(0, data.vocab_size - 1);
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                data.index[i] = dist_idx(gen_idx);
            }
    }

    // Initialize embedding gradients
    data.embed.resize(data.m * data.n * data.k);
    switch(strategy)
    {
        case DataGen::PRESET:
            for(Index i = 0; i < data.m * data.n * data.k; ++i)
            {
                data.embed[i] = Y(i + 1) / Y{1000};
            }
            break;
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(0.1, 2.0);
            for(Index i = 0; i < data.m * data.n * data.k; ++i)
            {
                data.embed[i] = dist(gen);
            }
    }

    // Initialize vocabulary gradients (zeros initially)
    data.vocab_init.resize(data.k_size * data.vocab_size);
    data.vocab.resize(data.k_size * data.vocab_size);
    data.vocab_ref.resize(data.k_size * data.vocab_size);
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index m,
    Index n,
    Index k,
    Index k_start,
    Index k_size,
    Index vocab_size,
    DataGen strategy
)
{
    TestData<T> data;
    data.m = m;
    data.n = n;
    data.k = k;
    data.k_start = k_start;
    data.k_size = k_size;
    data.vocab_size = vocab_size;

    // Generate data by a provided strategy
    generate_data(data, strategy);

    // Set accuracy threshold for each precision
    if (std::is_same_v<T, bf16_t>)
    {
        data.eps_check = 1e-1;
    }
    else if (std::is_same_v<T, fp16_t>)
    {
        data.eps_check = 1e-2;
    }
    else if (std::is_same_v<T, fp32_t>)
    {
        data.eps_check = 3.1e-3;
    }
    else if (std::is_same_v<T, fp64_t>)
    {
        data.eps_check = 1e-7;
    }
    else
    {
        throw std::runtime_error("Unsupported data type");
    }

    // Compute reference outputs
    reference_embedding_backward(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& vocab_out
)
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < data.k_size * data.vocab_size; ++i)
    {
        Y vocab_ref = static_cast<Y>(data.vocab_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(vocab_out[i]),
            WithinRel(vocab_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> vocab_cpu(data.vocab_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][embedding_backward][cpu][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "][k_size=" +
            std::to_string(data.k_size) +
            "]"
        )
        {
            cpu<T>(
                data.m,
                data.n,
                data.k,
                data.k_start,
                data.k_size,
                &data.index[0],
                &data.embed[0],
                &vocab_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.m,
            data.n,
            data.k,
            data.k_start,
            data.k_size,
            &data.index[0],
            &data.embed[0],
            &vocab_cpu[0]
        );
        verify_results(data, vocab_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_vocab, *dev_embed;
    int64_t *dev_index;

    CUDA_CHECK(cudaMalloc(&dev_vocab, sizeof(T) * data.k_size * data.vocab_size),
               "cudaMalloc dev_vocab");
    CUDA_CHECK(cudaMalloc(&dev_index, sizeof(int64_t) * data.m * data.n),
               "cudaMalloc dev_index");
    CUDA_CHECK(cudaMalloc(&dev_embed, sizeof(T) * data.m * data.n * data.k),
               "cudaMalloc dev_embed");

    std::vector<T> vocab_cuda(data.vocab);

    CUDA_CHECK(cudaMemcpy(dev_vocab, &data.vocab[0], sizeof(T) * data.k_size * data.vocab_size,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_vocab");
    CUDA_CHECK(cudaMemcpy(dev_index, &data.index[0], sizeof(int64_t) * data.m * data.n,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_index");
    CUDA_CHECK(cudaMemcpy(dev_embed, &data.embed[0], sizeof(T) * data.m * data.n * data.k,
                          cudaMemcpyHostToDevice), "cudaMemcpy dev_embed");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][embedding_backward][cuda][m=" +
            std::to_string(data.m) +
            "][n=" +
            std::to_string(data.n) +
            "][k=" +
            std::to_string(data.k) +
            "][k_size=" +
            std::to_string(data.k_size) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.m,
                data.n,
                data.k,
                data.k_start,
                data.k_size,
                dev_index,
                dev_embed,
                dev_vocab
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.m,
            data.n,
            data.k,
            data.k_start,
            data.k_size,
            dev_index,
            dev_embed,
            dev_vocab
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(&vocab_cuda[0], dev_vocab, sizeof(T) * data.k_size * data.vocab_size,
                              cudaMemcpyDeviceToHost), "cudaMemcpy vocab_cuda");

        verify_results(data, vocab_cuda);
    }

    CUDA_CHECK(cudaFree(dev_vocab), "cudaFree dev_vocab");
    CUDA_CHECK(cudaFree(dev_index), "cudaFree dev_index");
    CUDA_CHECK(cudaFree(dev_embed), "cudaFree dev_embed");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "EmbeddingBackward Kernel Verification",
    "[embedding_backward]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(2, 5);
    const Index n = GENERATE(3, 7);
    const Index k = GENERATE(4, 8);
    const Index k_start = GENERATE(0, 2);
    const Index k_size = GENERATE(2, 4);
    const Index vocab_size = GENERATE(10, 20);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        m,
        n,
        k,
        k_start,
        k_size,
        vocab_size,
        strategy
    );

    SECTION("cpu")
    {
        run_cpu_test<T, false>(data);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        run_cuda_test<T, false>(data);
    }
#endif
}

// Catch2-based benchmarks
TEMPLATE_TEST_CASE(
    "EmbeddingBackward Kernel Benchmark",
    "[embedding_backward][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index m = GENERATE(16, 32);
    const Index n = GENERATE(64, 128);
    const Index k = GENERATE(256, 512);
    const Index k_start = GENERATE(0);
    const Index k_size = GENERATE(128, 256);
    const Index vocab_size = GENERATE(1000, 5000);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        m,
        n,
        k,
        k_start,
        k_size,
        vocab_size,
        strategy
    );

    SECTION("cpu")
    {
        run_cpu_test<T, true>(data);
    }

#ifdef NNTILE_USE_CUDA
    SECTION("cuda")
    {
        run_cuda_test<T, true>(data);
    }
#endif
}
