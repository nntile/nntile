/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/embedding.cc
 * Embedding lookup operation for buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/embedding.hh"

// Standard libraries
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <cstdint>

// Third-party libraries
#include <catch2/catch_all.hpp>

// Other NNTile headers
// CUDA_CHECK definition
#include <nntile/kernel/cuda.hh>
#include <nntile/base_types.hh>

// Use namespaces for shorter code
using namespace Catch;
using namespace Catch::Matchers;

// Use tested NNTile namespaces
using namespace nntile;
using namespace nntile::kernel;
using namespace nntile::kernel::embedding;

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

    std::vector<nntile::int64_t> index_init;
    std::vector<T> vocab_init;
    std::vector<T> embed_init;

    std::vector<T> embed_ref;
};

// Reference implementation of the embedding operation
template<typename T>
void reference_embedding(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.m == 0 || data.n == 0 || data.k_size == 0)
    {
        return;
    }

    // Initialize output embedding tensor
    data.embed_ref = data.embed_init;

    // Update output embedding tensor
    for(Index i2 = 0; i2 < data.n; ++i2)
    {
        for(Index i1 = 0; i1 < data.m; ++i1)
        {
            auto idx = data.index_init[i2 * data.m + i1].value;
            // Input slice of vocabulary
            const T *vocab_slice = &data.vocab_init[data.k_size * idx];
            // Output slice to be updated
            Index embed_idx = (i2 * data.k + data.k_start) * data.m + i1;
            T *embed_slice = &data.embed_ref[embed_idx];

            for(Index i0 = 0; i0 < data.k_size; ++i0)
            {
                embed_slice[i0 * data.m] = vocab_slice[i0];
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

    data.index_init.resize(data.m * data.n);
    data.vocab_init.resize(data.k_size * data.vocab_size);
    data.embed_init.resize(data.m * data.n * data.k);
    data.embed_ref.resize(data.m * data.n * data.k);

    switch(strategy)
    {
        case DataGen::PRESET:
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                // Ensure indices are within vocab_size
                nntile::int64_t::repr_t idx = (i * 7 + 3) % data.vocab_size;
                data.index_init[i] = idx;
            }
            for(Index i = 0; i < data.k_size * data.vocab_size; ++i)
            {
                Y vocab_val = Y(i + 1) / Y(1000);
                data.vocab_init[i] = vocab_val;
            }
            for(Index i = 0; i < data.m * data.n * data.k; ++i)
            {
                Y embed_val = Y(2 * i + 3) / Y(2000);
                data.embed_init[i] = embed_val;
            }
            break;
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(0.1, 2.0);
            for(Index i = 0; i < data.k_size * data.vocab_size; ++i)
            {
                data.vocab_init[i] = dist(gen);
            }
            std::uniform_int_distribution<nntile::int64_t::repr_t>
                idx(0, data.vocab_size - 1);
            for(Index i = 0; i < data.m * data.n; ++i)
            {
                data.index_init[i] = idx(gen);
            }
            for(Index i = 0; i < data.m * data.n * data.k; ++i)
            {
                data.embed_init[i] = dist(gen);
            }
            break;
    }
}

// Get test input data (reference computation is done separately)
template<typename T>
TestData<T> get_test_input_data(
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

    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<nntile::int64_t>& index,
    const std::vector<T>& vocab,
    const std::vector<T>& embed
)
{
    // Check that index was not changed during kernel execution
    for(Index i = 0; i < data.m * data.n; ++i)
    {
        REQUIRE(index[i].value == data.index_init[i].value);
    }

    // Check that vocab was not changed during kernel execution
    for(Index i = 0; i < data.k_size * data.vocab_size; ++i)
    {
        REQUIRE(vocab[i].value == data.vocab_init[i].value);
    }

    // Check that embed (output) matches reference
    for(Index i = 0; i < data.m * data.n * data.k; ++i)
    {
        REQUIRE(embed[i].value == data.embed_ref[i].value);
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> embed_cpu(data.embed_init);
    std::vector<nntile::int64_t> index_cpu(data.index_init);
    std::vector<T> vocab_cpu(data.vocab_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][embedding][cpu][m=" +
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
                &index_cpu[0],
                &vocab_cpu[0],
                &embed_cpu[0]
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
            &index_cpu[0],
            &vocab_cpu[0],
            &embed_cpu[0]
        );
        verify_results(data, index_cpu, vocab_cpu, embed_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_vocab, *dev_embed;
    nntile::int64_t *dev_index;

    CUDA_CHECK(
        cudaMalloc(
            &dev_vocab,
            sizeof(T) * data.vocab_init.size()
        ),
        "cudaMalloc dev_vocab"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_index,
            sizeof(nntile::int64_t) * data.index_init.size()
        ),
        "cudaMalloc dev_index"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_embed,
            sizeof(T) * data.embed_init.size()
        ),
        "cudaMalloc dev_embed"
    );

    std::vector<T> embed_cuda(data.embed_init);
    std::vector<nntile::int64_t> index_cuda(data.index_init);
    std::vector<T> vocab_cuda(data.vocab_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_vocab,
            &data.vocab_init[0],
            sizeof(T) * data.vocab_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_vocab"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_index,
            &index_cuda[0],
            sizeof(nntile::int64_t) * data.index_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_index"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_embed,
            &embed_cuda[0],
            sizeof(T) * data.embed_init.size(),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_embed"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][embedding][cuda][m=" +
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
                dev_vocab,
                dev_embed
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
            dev_vocab,
            dev_embed
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &embed_cuda[0],
                dev_embed,
                sizeof(T) * data.embed_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy embed_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &index_cuda[0],
                dev_index,
                sizeof(nntile::int64_t) * data.index_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy index_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &vocab_cuda[0],
                dev_vocab,
                sizeof(T) * data.vocab_init.size(),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy vocab_cuda"
        );

        verify_results(data, index_cuda, vocab_cuda, embed_cuda);
    }

    CUDA_CHECK(cudaFree(dev_vocab), "cudaFree dev_vocab");
    CUDA_CHECK(cudaFree(dev_index), "cudaFree dev_index");
    CUDA_CHECK(cudaFree(dev_embed), "cudaFree dev_embed");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Embedding Kernel Verification",
    "[embedding]",
    fp64_t,
    fp32_t,
    bf16_t,
    fp16_t
)
{
    using T = TestType;
    const Index m = GENERATE(2, 5);
    const Index n = GENERATE(3, 7);
    const Index k = GENERATE(6, 8);
    const Index k_start = GENERATE(0, 2);
    const Index k_size = GENERATE(2, 4);
    const Index vocab_size = GENERATE(10, 20);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(
        m,
        n,
        k,
        k_start,
        k_size,
        vocab_size,
        strategy
    );

    // Compute reference outputs for verification
    reference_embedding(data);

    SECTION(("cpu")
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
    "Embedding Kernel Benchmark",
    "[embedding][!benchmark]",
    fp64_t,
    fp32_t,
    bf16_t,
    fp16_t
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

    auto data = get_test_input_data<T>(
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
