/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/addcdiv.cc
 * Per-element addcdiv operation for buffers
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/addcdiv.hh"

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
using namespace nntile::kernel::addcdiv;

// Type to acquire reference values
using ref_t = double;

// Struct to hold test data and reference results
template<typename T>
struct TestData
{
    using Y = typename T::repr_t;
    Index num_elems; // Number of data elements
    Scalar val;
    Scalar eps;

    Y eps_check;

    std::vector<T> nom_init;
    std::vector<T> denom_init;
    std::vector<T> src_init;

    std::vector<T> src_ref;
};

// Reference implementation of the addcdiv operation
template<typename T>
void reference_addcdiv(TestData<T>& data)
{
    using Y = typename T::repr_t;
    if (data.num_elems == 0)
    {
        return;
    }
    const ref_t val_r = data.val;
    const ref_t eps_r = data.eps;

    for(Index i = 0; i < data.num_elems; ++i)
    {
        ref_t src_val = static_cast<Y>(data.src_init[i]);
        ref_t nom_val = static_cast<Y>(data.nom_init[i]);
        ref_t denom_val = static_cast<Y>(data.denom_init[i]);
        ref_t result = src_val + val_r * nom_val / (denom_val + eps_r);
        data.src_ref[i] = static_cast<Y>(result);
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
void generate_data(TestData<T>& data, Index num_elems, DataGen strategy)
{
    using Y = typename T::repr_t;
    data.num_elems = num_elems;

    data.nom_init.resize(num_elems);
    data.denom_init.resize(num_elems);
    data.src_init.resize(num_elems);
    data.src_ref.resize(num_elems);

    switch(strategy)
    {
        // Non-random input generation
        case DataGen::PRESET:
            {
                ref_t sign_factor = -1.;
                for(Index i = 0; i < num_elems; ++i)
                {
                    Y src_val = 2 * i + 1 - num_elems;
                    data.src_init[i] = src_val * sign_factor;
                    Y nom_val = num_elems - i;
                    data.nom_init[i] = nom_val;
                    Y denom_val = i + 1;
                    data.denom_init[i] = denom_val;
                    sign_factor = -sign_factor;
                }
            }
            break;
        // Specific random initialization
        case DataGen::RANDOM:
            std::mt19937 gen(42);
            std::uniform_real_distribution<Y> dist(1.0, 2.0);
            for(Index i = 0; i < num_elems; ++i)
            {
                data.src_init[i] = dist(gen);
                data.nom_init[i] = dist(gen);
                // Avoid division by zero in reference
                data.denom_init[i] = 0.5 * dist(gen) + 0.1;
            }
    }
}

// Get test data and reference results
template<typename T>
TestData<T> get_test_data(
    Index num_elems,
    Scalar val,
    Scalar eps,
    DataGen strategy
)
{
    TestData<T> data;
    // Generate data by a provided strategy
    generate_data(data, num_elems, strategy);
    // Fill in remaining fields of TestData
    data.val = val;
    data.eps = eps;
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
    reference_addcdiv(data);
    return data;
}

// Helper function to verify results
template<typename T>
void verify_results(
    const TestData<T>& data,
    const std::vector<T>& nom,
    const std::vector<T>& denom,
    const std::vector<T>& src
)
{
    using Y = typename T::repr_t;

    // Check that nom was not changed during kernel execution
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y nom_val = static_cast<Y>(nom[i]);
        Y nom_init_val = static_cast<Y>(data.nom_init[i]);
        REQUIRE(nom_val == nom_init_val);
    }

    // Check that denom was not changed during kernel execution
    for(Index i = 0; i < data.num_elems; ++i)
    {
        Y denom_val = static_cast<Y>(denom[i]);
        Y denom_init_val = static_cast<Y>(data.denom_init[i]);
        REQUIRE(denom_val == denom_init_val);
    }

    // Check that src (output) matches reference
    for(Index i = 0; i < data.num_elems; ++i)
    {
        const Y src_ref = static_cast<Y>(data.src_ref[i]);
        REQUIRE_THAT(
            static_cast<Y>(src[i]),
            WithinRel(src_ref, data.eps_check)
        );
    }
}

// Helper function to run CPU test and verify results
template<typename T, bool run_bench>
void run_cpu_test(TestData<T>& data)
{
    std::vector<T> src_cpu(data.src_init);
    std::vector<T> nom_cpu(data.nom_init);
    std::vector<T> denom_cpu(data.denom_init);

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][addcdiv][cpu][nelems=" +
            std::to_string(data.num_elems) +
            "][val=" +
            std::to_string(data.val) +
            "][eps=" +
            std::to_string(data.eps) +
            "]"
        )
        {
            cpu<T>(
                data.val,
                data.eps,
                data.num_elems,
                &nom_cpu[0],
                &denom_cpu[0],
                &src_cpu[0]
            );
        };
    }
    else
    {
        cpu<T>(
            data.val,
            data.eps,
            data.num_elems,
            &nom_cpu[0],
            &denom_cpu[0],
            &src_cpu[0]
        );
        verify_results(data, nom_cpu, denom_cpu, src_cpu);
    }
}

#ifdef NNTILE_USE_CUDA

// Helper function to run CUDA test and verify results
template<typename T, bool run_bench>
void run_cuda_test(TestData<T>& data)
{
    T *dev_src, *dev_nom, *dev_denom;
    CUDA_CHECK(
        cudaMalloc(
            &dev_src,
            sizeof(T) * data.num_elems
        ),
        "cudaMalloc dev_src"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_nom,
            sizeof(T) * data.num_elems
        ),
        "cudaMalloc dev_nom"
    );
    CUDA_CHECK(
        cudaMalloc(
            &dev_denom,
            sizeof(T) * data.num_elems
        ),
        "cudaMalloc dev_denom"
    );

    std::vector<T> src_cuda(data.src_init);
    std::vector<T> nom_cuda(data.nom_init);
    std::vector<T> denom_cuda(data.denom_init);

    CUDA_CHECK(
        cudaMemcpy(
            dev_src,
            &data.src_init[0],
            sizeof(T) * data.num_elems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_src"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_nom,
            &data.nom_init[0],
            sizeof(T) * data.num_elems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_nom"
    );
    CUDA_CHECK(
        cudaMemcpy(
            dev_denom,
            &data.denom_init[0],
            sizeof(T) * data.num_elems,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy dev_denom"
    );

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr (run_bench)
    {
        BENCHMARK(
            "[kernel][addcdiv][cuda][nelems=" +
            std::to_string(data.num_elems) +
            "][val=" +
            std::to_string(data.val) +
            "][eps=" +
            std::to_string(data.eps) +
            "]"
        )
        {
            cuda<T>(
                stream,
                data.val,
                data.eps,
                data.num_elems,
                dev_nom,
                dev_denom,
                dev_src
            );
            cudaStreamSynchronize(stream);
        };
    }
    else
    {
        cuda<T>(
            stream,
            data.val,
            data.eps,
            data.num_elems,
            dev_nom,
            dev_denom,
            dev_src
        );
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(
            cudaMemcpy(
                &src_cuda[0],
                dev_src,
                sizeof(T) * data.num_elems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy src_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &nom_cuda[0],
                dev_nom,
                sizeof(T) * data.num_elems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy nom_cuda"
        );
        CUDA_CHECK(
            cudaMemcpy(
                &denom_cuda[0],
                dev_denom,
                sizeof(T) * data.num_elems,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy denom_cuda"
        );

        verify_results(data, nom_cuda, denom_cuda, src_cuda);
    }

    CUDA_CHECK(cudaFree(dev_src), "cudaFree dev_src");
    CUDA_CHECK(cudaFree(dev_nom), "cudaFree dev_nom");
    CUDA_CHECK(cudaFree(dev_denom), "cudaFree dev_denom");
    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
}
#endif

// Catch2-based tests
TEMPLATE_TEST_CASE(
    "Addcdiv Kernel Verification",
    "[addcdiv]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(5, 129);
    const Scalar val = GENERATE(-5.0, 0.0, 1.0, 2.5);
    const Scalar eps = GENERATE(0.0, 1e-5, 1e-2);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_data<T>(
        num_elems,
        val,
        eps,
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
    "Addcdiv Kernel Benchmark",
    "[addcdiv][!benchmark]",
    fp64_t,
    fp32_t,
    fp16_t,
    bf16_t
)
{
    using T = TestType;
    const Index num_elems = GENERATE(512, 1024*1024, 4096*16384);
    const Scalar val = GENERATE(1.0);
    const Scalar eps = GENERATE(1e-5);
    const DataGen strategy = GENERATE(DataGen::PRESET);

    auto data = get_test_data<T>(
        num_elems,
        val,
        eps,
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
