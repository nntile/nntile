/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/sdpa.cc
 * Tests for SDPA module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/sdpa.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

TEST_CASE("Sdpa Vanilla Constructor", "[module]")
{
    NNGraph g("sdpa");

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    REQUIRE(sdpa.flash_attention() == false);
    REQUIRE(sdpa.head_size() == 64);
    REQUIRE(sdpa.batch_ndim() == 2);
    REQUIRE(sdpa.scale() > 0);
}

TEST_CASE("Sdpa Flash Constructor", "[module]")
{
    NNGraph g("sdpa");

    Sdpa sdpa(g, "sdpa", 64, 2, true, DataType::FP16);
    REQUIRE(sdpa.flash_attention() == true);
    REQUIRE(sdpa.head_size() == 64);
}

TEST_CASE("Sdpa ConstructorValidations", "[module]")
{
    NNGraph g("sdpa");

    REQUIRE_THROWS_AS(
        Sdpa(g, "sdpa", 0, 2, DataType::FP32),
        std::invalid_argument);

    REQUIRE_THROWS_AS(
        Sdpa(g, "sdpa", 64, 2, false, DataType::FP16),
        std::invalid_argument);

    REQUIRE_THROWS_AS(
        Sdpa(g, "sdpa", 64, 2, true, DataType::FP32),
        std::invalid_argument);
}

TEST_CASE("Sdpa Vanilla BuildForward", "[module]")
{
    NNGraph g("sdpa");

    // Layout: [head_size, n_seq, batch...]
    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto& k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    auto& output = sdpa.build_forward(q, k, v, nullptr);

    REQUIRE(output.shape() == std::vector<Index>({64, 8, 2, 4}));
    REQUIRE(output.name() == "sdpa_output");

    size_t gemm_count = 0;
    size_t maxsumexp_count = 0;
    size_t softmax_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::GEMM)
        {
            ++gemm_count;
        }
        if(op->type() == OpType::MAXSUMEXP)
        {
            ++maxsumexp_count;
        }
        if(op->type() == OpType::SOFTMAX_INPLACE)
        {
            ++softmax_count;
        }
    }
    REQUIRE(gemm_count == 2);
    REQUIRE(maxsumexp_count == 1);
    REQUIRE(softmax_count == 1);
}

TEST_CASE("Sdpa Vanilla BuildForwardWithMask", "[module]")
{
    NNGraph g("sdpa");

    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto& k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);
    auto& mask = g.tensor({8, 8}, "mask", DataType::BOOL);

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    auto& output = sdpa.build_forward(q, k, v, &mask);

    REQUIRE(output.shape() == std::vector<Index>({64, 8, 2, 4}));

    size_t mask_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::MASK_SCALAR)
        {
            ++mask_count;
        }
    }
    REQUIRE(mask_count == 1);
}

TEST_CASE("Sdpa Vanilla BuildBackward", "[module]")
{
    NNGraph g("sdpa");

    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto& k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    auto& output = sdpa.build_forward(q, k, v, nullptr);

    g.get_or_create_grad(output, "output_grad");
    sdpa.build_backward();

    REQUIRE(q.grad() != nullptr);
    REQUIRE(k.grad() != nullptr);
    REQUIRE(v.grad() != nullptr);
    REQUIRE(q.grad()->shape() == q.shape());
    REQUIRE(k.grad()->shape() == k.shape());
    REQUIRE(v.grad()->shape() == v.shape());
}

TEST_CASE("Sdpa Vanilla BuildBackwardRequiresForward", "[module]")
{
    NNGraph g("sdpa");

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    REQUIRE_THROWS_AS(sdpa.build_backward(), std::runtime_error);
}

TEST_CASE("Sdpa Vanilla BuildForwardValidatesShape", "[module]")
{
    NNGraph g("sdpa");

    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto& k = g.tensor({32, 8, 2, 4}, "k", DataType::FP32);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    Sdpa sdpa(g, "sdpa", 64, 2, DataType::FP32);
    REQUIRE_THROWS_AS(
        sdpa.build_forward(q, k, v, nullptr),
        std::invalid_argument);
}

TEST_CASE("Sdpa Flash BuildForward", "[module]")
{
    NNGraph g("sdpa");

    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP16);
    auto& k = g.tensor({64, 8, 2, 4}, "k", DataType::FP16);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP16);

    Sdpa sdpa(g, "sdpa", 64, 2, true, DataType::FP16);
    auto& output = sdpa.build_forward(q, k, v, nullptr);

    REQUIRE(output.shape() == std::vector<Index>({64, 8, 2, 4}));

    size_t flash_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->type() == OpType::FLASH_SDPA_FWD_CUDNN)
        {
            ++flash_count;
        }
    }
    REQUIRE(flash_count == 1);
}

TEST_CASE("Sdpa Flash BuildForwardValidatesDtype", "[module]")
{
    NNGraph g("sdpa");

    auto& q = g.tensor({64, 8, 2, 4}, "q", DataType::FP32);
    auto& k = g.tensor({64, 8, 2, 4}, "k", DataType::FP32);
    auto& v = g.tensor({64, 8, 2, 4}, "v", DataType::FP32);

    Sdpa sdpa(g, "sdpa", 64, 2, true, DataType::FP16);
    REQUIRE_THROWS_AS(
        sdpa.build_forward(q, k, v, nullptr),
        std::invalid_argument);
}
