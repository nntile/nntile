/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/tile/execution_driver.cc
 * ExecutionDriver runs a TileGraph; it does not lower TensorGraph.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"
#include "test_frobenius.hh"

#include <nntile/execution_driver.hh>
#include <nntile/tile.hh>
#include <nntile/tile/ops/add.hh>
#include <nntile/tile/ops/add_inplace.hh>
#include <nntile/tile/ops/copy.hh>
#include <nntile/tile/ops/embedding.hh>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace nntile;
namespace tg = nntile::tile;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RuntimeExecutionDriver submit does not lower TensorGraph",
    "[graph][tile][driver]")
{
    std::vector<Index> shape = {4};
    TileGraph graph("driver_exec");
    auto *x = graph.data(shape, "x", DataType::FP32);
    auto *y = graph.data(shape, "y", DataType::FP32);
    tg::add_inplace(Scalar(2.0), x, Scalar(1.0), y);

    RuntimeExecutionDriver driver(graph);
    driver.runtime().compile();
    driver.runtime().bind_data(x, std::vector<float>{1, 2, 3, 4});
    driver.runtime().bind_data(y, std::vector<float>{10, 20, 30, 40});
    driver.submit(graph);
    driver.wait();

    const auto result = driver.runtime().get_output<float>(y);
    const std::vector<float> expected{12.f, 24.f, 36.f, 48.f};
    nntile::test::require_relative_element_error(result, expected);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RuntimeExecutionDriver two submits share one wait",
    "[graph][tile][driver]")
{
    std::vector<Index> shape = {2};
    TileGraph graph("driver_two_submit");
    auto *a = graph.data(shape, "a", DataType::FP32);
    auto *b = graph.data(shape, "b", DataType::FP32);
    tg::add_inplace(Scalar(1.0), a, Scalar(1.0), b);

    RuntimeExecutionDriver driver(graph);
    driver.runtime().compile();
    driver.runtime().bind_data(a, std::vector<float>{1, 2});
    driver.runtime().bind_data(b, std::vector<float>{3, 4});
    driver.submit(graph);

    auto *c = graph.data(shape, "c", DataType::FP32);
    tg::add_inplace(Scalar(1.0), b, Scalar(1.0), c);
    driver.runtime().compile();
    driver.runtime().bind_data(c, std::vector<float>{0, 0});
    driver.submit(graph);
    driver.wait();

    const auto b_out = driver.runtime().get_output<float>(b);
    const auto c_out = driver.runtime().get_output<float>(c);
    nntile::test::require_relative_element_error(b_out, {4.f, 6.f});
    nntile::test::require_relative_element_error(c_out, {4.f, 6.f});
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RuntimeExecutionDriver rejects a different TileGraph",
    "[graph][tile][driver]")
{
    TileGraph a("driver_a");
    TileGraph b("driver_b");
    RuntimeExecutionDriver driver(a);
    REQUIRE_THROWS_AS(driver.submit(b), std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RuntimeExecutionDriver INT64 bind and gather",
    "[graph][tile][driver]")
{
    TileGraph graph("driver_int64");
    auto *src = graph.data({4}, "src", DataType::INT64);
    auto *dst = graph.data({4}, "dst", DataType::INT64);
    tg::copy(src, dst);

    RuntimeExecutionDriver driver(graph);
    driver.runtime().compile();
    driver.runtime().bind_data(
        src, std::vector<std::int64_t>{1, 2, 3, 4});
    driver.runtime().bind_data(
        dst, std::vector<std::int64_t>{0, 0, 0, 0});
    driver.submit(graph);
    driver.wait();
    auto const out = driver.runtime().get_output<std::int64_t>(dst);
    REQUIRE(out == std::vector<std::int64_t>{1, 2, 3, 4});
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "RuntimeExecutionDriver EMBEDDING mixed INT64/FP32",
    "[graph][tile][driver]")
{
    Index const m = 2, n = 2, k = 3, k0 = 0, ks = 3;
    TileGraph graph("driver_emb");
    auto *index = graph.data({m, n}, "index", DataType::INT64);
    auto *vocab = graph.data({ks, 5}, "vocab", DataType::FP32);
    auto *embed = graph.data({m, k, n}, "embed", DataType::FP32);
    tg::embedding(m, n, k, k0, ks, index, vocab, embed);

    std::vector<float> voc(15);
    for (int i = 0; i < 15; ++i)
    {
        voc[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }
    RuntimeExecutionDriver driver(graph);
    driver.runtime().compile();
    driver.runtime().bind_data(
        index, std::vector<std::int64_t>{0, 2, 4, 1});
    driver.runtime().bind_data(vocab, voc);
    driver.runtime().bind_data(
        embed, std::vector<float>(12, 0.f));
    driver.submit(graph);
    driver.wait();
    auto const out = driver.runtime().get_output<float>(embed);
    REQUIRE(out.size() == 12);
    bool nonzero = false;
    for (float v : out)
    {
        if (v != 0.f)
        {
            nonzero = true;
            break;
        }
    }
    REQUIRE(nonzero);
}
