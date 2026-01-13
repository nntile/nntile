/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/kernel/accumulate_attn_output.cc
 * Accumulate attention outputs kernel tests
 *
 * @version 1.1.0
 * */

// Corresponding header
#include "nntile/kernel/accumulate_attn_output.hh"

// Standard libraries
#include <vector>
#include <limits>
#include <random>
#include <cmath>
#include <type_traits>
#include <algorithm>
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
using namespace nntile::kernel::accumulate_attn_output;

namespace
{

enum class DataGen
{
    PRESET,
    RANDOM
};

template<typename T>
struct TestData
{
    using repr_t = typename T::repr_t;
    using lse_t = typename fp32_t::repr_t;

    Index head{};
    Index seq{};
    Index batch{};
    Index nelems{};
    Index attn_nelems{};
    lse_t lse_tol{};
    lse_t attn_tol{};

    std::vector<fp32_t> src_lse_init;
    std::vector<fp32_t> dst_lse_init;
    std::vector<fp32_t> dst_lse_ref;

    std::vector<T> src_attn_init;
    std::vector<T> dst_attn_init;
    std::vector<T> dst_attn_ref;
};

template<typename T>
void generate_data(TestData<T> &data, Index head, Index seq, Index batch,
        DataGen strategy)
{
    using repr_t = typename T::repr_t;
    using lse_t = typename fp32_t::repr_t;
    data.head = head;
    data.seq = seq;
    data.batch = batch;
    data.nelems = seq * batch;
    data.attn_nelems = data.nelems * head;

    data.src_lse_init.resize(data.nelems);
    data.dst_lse_init.resize(data.nelems);
    data.dst_lse_ref.resize(data.nelems);
    data.src_attn_init.resize(data.attn_nelems);
    data.dst_attn_init.resize(data.attn_nelems);
    data.dst_attn_ref.resize(data.attn_nelems);

    const lse_t neg_inf = -std::numeric_limits<lse_t>::infinity();

    switch(strategy)
    {
        case DataGen::PRESET:
            for(Index b = 0; b < data.batch; ++b)
            {
                for(Index s = 0; s < data.seq; ++s)
                {
                    const Index linear = b * data.seq + s;
                    const lse_t base =
                            lse_t(-3.0f) + lse_t(0.25f) * lse_t(linear % 17);
                    const bool dst_active = ((linear + 1) % 3) != 0;
                    const bool src_active = ((linear + 2) % 4) != 0;
                    data.dst_lse_init[linear] =
                            dst_active ? base : neg_inf;
                    data.src_lse_init[linear] =
                            src_active ? (base + lse_t(0.5f)) : neg_inf;

                    for(Index h = 0; h < data.head; ++h)
                    {
                        const Index attn_idx = linear * data.head + h;
                        const repr_t dst_val = dst_active
                                ? repr_t(0.1f * ((linear + h) % 11) - 1.0f)
                                : repr_t(0);
                        const repr_t src_val = src_active
                                ? repr_t(0.2f * ((linear + h + 3) % 13) - 0.5f)
                                : repr_t(0);
                        data.dst_attn_init[attn_idx] =
                                static_cast<T>(dst_val);
                        data.src_attn_init[attn_idx] =
                                static_cast<T>(src_val);
                    }
                }
            }
            break;
        case DataGen::RANDOM:
        {
            std::mt19937 gen(static_cast<unsigned>(seq * 37 + batch * 53 + 11));
            std::uniform_real_distribution<float> dist_val(-2.0f, 2.0f);
            std::uniform_real_distribution<float> dist_lse(-5.0f, 5.0f);
            std::bernoulli_distribution zero_prob(0.3);
            for(Index b = 0; b < data.batch; ++b)
            {
                for(Index s = 0; s < data.seq; ++s)
                {
                    const Index linear = b * data.seq + s;
                    const bool dst_has_mass = !zero_prob(gen);
                    const bool src_has_mass = !zero_prob(gen);

                    data.dst_lse_init[linear] =
                            dst_has_mass ? dist_lse(gen) : neg_inf;
                    data.src_lse_init[linear] =
                            src_has_mass ? dist_lse(gen) : neg_inf;

                    for(Index h = 0; h < data.head; ++h)
                    {
                        const Index attn_idx = linear * data.head + h;
                        const repr_t dst_val = dst_has_mass
                                ? repr_t(dist_val(gen)) : repr_t(0);
                        const repr_t src_val = src_has_mass
                                ? repr_t(dist_val(gen)) : repr_t(0);
                        data.dst_attn_init[attn_idx] =
                                static_cast<T>(dst_val);
                        data.src_attn_init[attn_idx] =
                                static_cast<T>(src_val);
                    }
                }
            }
            break;
        }
    }

    data.dst_lse_ref = data.dst_lse_init;
    data.dst_attn_ref = data.dst_attn_init;
}

template<typename T>
TestData<T> get_test_input_data(Index head, Index seq, Index batch, DataGen strategy)
{
    TestData<T> data;
    generate_data(data, head, seq, batch, strategy);
    data.lse_tol = typename fp32_t::repr_t(1e-5f);
    if(std::is_same_v<T, fp32_t>)
    {
        data.attn_tol = typename fp32_t::repr_t(1e-5f);
    }
    else
    {
        data.attn_tol = typename fp32_t::repr_t(5e-3f);
    }
    return data;
}

template<typename T>
void reference_accumulate(TestData<T> &data)
{
    using repr_t = typename T::repr_t;
    using ref_t = double;
    using lse_t = typename fp32_t::repr_t;

    for(Index b = 0; b < data.batch; ++b)
    {
        for(Index s = 0; s < data.seq; ++s)
        {
            const Index lse_idx = b * data.seq + s;
            const lse_t old_lse =
                    static_cast<lse_t>(data.dst_lse_ref[lse_idx]);
            const lse_t incoming_lse =
                    static_cast<lse_t>(data.src_lse_init[lse_idx]);

            const bool dst_active = !(std::isinf(old_lse) && old_lse < 0);
            const bool src_active = !(std::isinf(incoming_lse) && incoming_lse < 0);

            if(!dst_active && !src_active)
            {
                continue;
            }

            lse_t new_lse;
            if(dst_active && src_active)
            {
                const lse_t max_lse = std::max(old_lse, incoming_lse);
                const lse_t sum = std::exp(old_lse - max_lse)
                        + std::exp(incoming_lse - max_lse);
                new_lse = max_lse + std::log(sum);
            }
            else
            {
                new_lse = dst_active ? old_lse : incoming_lse;
            }

            const lse_t dst_weight =
                    dst_active ? std::exp(old_lse - new_lse) : lse_t(0);
            const lse_t src_weight =
                    src_active ? std::exp(incoming_lse - new_lse) : lse_t(0);

            data.dst_lse_ref[lse_idx] = new_lse;

            for(Index h = 0; h < data.head; ++h)
            {
                const Index attn_idx = lse_idx * data.head + h;
                const ref_t dst_val =
                        static_cast<repr_t>(data.dst_attn_ref[attn_idx]);
                const ref_t src_val =
                        static_cast<repr_t>(data.src_attn_init[attn_idx]);
                const ref_t updated = dst_weight * dst_val + src_weight * src_val;

                data.dst_attn_ref[attn_idx] =
                        static_cast<T>(static_cast<repr_t>(updated));
            }
        }
    }
}

template<typename T>
void verify_results(const TestData<T> &data,
        const std::vector<fp32_t> &src_lse,
        const std::vector<T> &src_attn,
        const std::vector<fp32_t> &dst_lse,
        const std::vector<T> &dst_attn)
{
    using repr_t = typename T::repr_t;
    using lse_t = typename fp32_t::repr_t;

    for(Index i = 0; i < data.nelems; ++i)
    {
        REQUIRE(static_cast<lse_t>(src_lse[i]) ==
                static_cast<lse_t>(data.src_lse_init[i]));
    }

    for(Index i = 0; i < data.attn_nelems; ++i)
    {
        REQUIRE(static_cast<repr_t>(src_attn[i]) ==
                static_cast<repr_t>(data.src_attn_init[i]));
    }

    for(Index i = 0; i < data.nelems; ++i)
    {
        const lse_t ref_lse = static_cast<lse_t>(data.dst_lse_ref[i]);
        const lse_t got_lse = static_cast<lse_t>(dst_lse[i]);
        if(std::isinf(ref_lse) && ref_lse < 0)
        {
            REQUIRE(std::isinf(got_lse));
            REQUIRE(got_lse < 0);
        }
        else
        {
            REQUIRE(got_lse ==
                    Catch::Approx(ref_lse).margin(static_cast<double>(data.lse_tol)));
        }
    }

    for(Index i = 0; i < data.attn_nelems; ++i)
    {
        const double ref_val =
                static_cast<double>(static_cast<repr_t>(data.dst_attn_ref[i]));
        const double got_val =
                static_cast<double>(static_cast<repr_t>(dst_attn[i]));
        REQUIRE(got_val ==
                Catch::Approx(ref_val).margin(static_cast<double>(data.attn_tol)));
    }
}

template<typename T, bool run_bench>
void run_cpu_test(TestData<T> &data)
{
    if constexpr(run_bench)
    {
        BENCHMARK(
                "[kernel][accumulate_attn_output][cpu][seq="
                + std::to_string(data.seq)
                + "][batch=" + std::to_string(data.batch)
                + "][head=" + std::to_string(data.head) + "]")
        {
            auto src_lse = data.src_lse_init;
            auto src_attn = data.src_attn_init;
            auto dst_lse = data.dst_lse_init;
            auto dst_attn = data.dst_attn_init;

            cpu<T>(data.head, data.seq, data.batch,
                    src_lse.data(), src_attn.data(),
                    dst_lse.data(), dst_attn.data());
        };
    }
    else
    {
        auto src_lse = data.src_lse_init;
        auto src_attn = data.src_attn_init;
        auto dst_lse = data.dst_lse_init;
        auto dst_attn = data.dst_attn_init;

        cpu<T>(data.head, data.seq, data.batch,
                src_lse.data(), src_attn.data(),
                dst_lse.data(), dst_attn.data());

        verify_results(data, src_lse, src_attn, dst_lse, dst_attn);
    }
}

#ifdef NNTILE_USE_CUDA
template<typename T, bool run_bench>
void run_cuda_test(TestData<T> &data)
{
    auto src_lse = data.src_lse_init;
    auto src_attn = data.src_attn_init;
    auto dst_lse = data.dst_lse_init;
    auto dst_attn = data.dst_attn_init;

    fp32_t *dev_src_lse = nullptr;
    T *dev_src_attn = nullptr;
    fp32_t *dev_dst_lse = nullptr;
    T *dev_dst_attn = nullptr;

    CUDA_CHECK(cudaMalloc(&dev_src_lse, sizeof(fp32_t) * data.nelems),
            "cudaMalloc dev_src_lse");
    CUDA_CHECK(cudaMalloc(&dev_src_attn, sizeof(T) * data.attn_nelems),
            "cudaMalloc dev_src_attn");
    CUDA_CHECK(cudaMalloc(&dev_dst_lse, sizeof(fp32_t) * data.nelems),
            "cudaMalloc dev_dst_lse");
    CUDA_CHECK(cudaMalloc(&dev_dst_attn, sizeof(T) * data.attn_nelems),
            "cudaMalloc dev_dst_attn");

    CUDA_CHECK(cudaMemcpy(dev_src_lse, src_lse.data(),
            sizeof(fp32_t) * data.nelems, cudaMemcpyHostToDevice),
            "cudaMemcpy dev_src_lse");
    CUDA_CHECK(cudaMemcpy(dev_src_attn, src_attn.data(),
            sizeof(T) * data.attn_nelems, cudaMemcpyHostToDevice),
            "cudaMemcpy dev_src_attn");
    CUDA_CHECK(cudaMemcpy(dev_dst_lse, dst_lse.data(),
            sizeof(fp32_t) * data.nelems, cudaMemcpyHostToDevice),
            "cudaMemcpy dev_dst_lse");
    CUDA_CHECK(cudaMemcpy(dev_dst_attn, dst_attn.data(),
            sizeof(T) * data.attn_nelems, cudaMemcpyHostToDevice),
            "cudaMemcpy dev_dst_attn");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    if constexpr(run_bench)
    {
        BENCHMARK(
                "[kernel][accumulate_attn_output][cuda][seq="
                + std::to_string(data.seq)
                + "][batch=" + std::to_string(data.batch)
                + "][head=" + std::to_string(data.head) + "]")
        {
            cuda<T>(stream, data.head, data.seq, data.batch,
                    dev_src_lse, dev_src_attn,
                    dev_dst_lse, dev_dst_attn);
            CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        };
    }
    else
    {
        cuda<T>(stream, data.head, data.seq, data.batch,
                dev_src_lse, dev_src_attn,
                dev_dst_lse, dev_dst_attn);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpy(dst_lse.data(), dev_dst_lse,
                sizeof(fp32_t) * data.nelems, cudaMemcpyDeviceToHost),
                "cudaMemcpy dst_lse");
        CUDA_CHECK(cudaMemcpy(dst_attn.data(), dev_dst_attn,
                sizeof(T) * data.attn_nelems, cudaMemcpyDeviceToHost),
                "cudaMemcpy dst_attn");
        CUDA_CHECK(cudaMemcpy(src_lse.data(), dev_src_lse,
                sizeof(fp32_t) * data.nelems, cudaMemcpyDeviceToHost),
                "cudaMemcpy src_lse");
        CUDA_CHECK(cudaMemcpy(src_attn.data(), dev_src_attn,
                sizeof(T) * data.attn_nelems, cudaMemcpyDeviceToHost),
                "cudaMemcpy src_attn");

        verify_results(data, src_lse, src_attn, dst_lse, dst_attn);
    }

    CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
    CUDA_CHECK(cudaFree(dev_src_lse), "cudaFree dev_src_lse");
    CUDA_CHECK(cudaFree(dev_src_attn), "cudaFree dev_src_attn");
    CUDA_CHECK(cudaFree(dev_dst_lse), "cudaFree dev_dst_lse");
    CUDA_CHECK(cudaFree(dev_dst_attn), "cudaFree dev_dst_attn");
}
#endif

} // namespace

TEMPLATE_TEST_CASE("Accumulate attention output kernel verification",
        "[kernel][accumulate_attn_output]",
        fp32_t, fp16_t, bf16_t)
{
    using T = TestType;
    // Include head sizes above warp size to exercise CUDA warp-stride loop.
    const Index head = GENERATE(1, 5, 33, 64);
    const Index seq = GENERATE(1, 3, 9);
    const Index batch = GENERATE(1, 5, 7);
    const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>(head, seq, batch, strategy);
    reference_accumulate(data);

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

TEMPLATE_TEST_CASE("Accumulate attention output kernel benchmark",
        "[kernel][accumulate_attn_output][!benchmark]",
        fp32_t, fp16_t, bf16_t)
{
    using T = TestType;
    const Index head = GENERATE(32, 64);
    const Index seq = GENERATE(16, 64, 256);
    const Index batch = GENERATE(16, 64, 256);

    auto data = get_test_input_data<T>(head, seq, batch, DataGen::PRESET);

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
