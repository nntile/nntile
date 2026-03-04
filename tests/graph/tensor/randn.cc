/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/randn.cc
 * Test TensorGraph randn operation against nntile::tensor::randn.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/randn.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

namespace
{

constexpr unsigned long long seed = 42;
constexpr Scalar mean = 0.0;
constexpr Scalar stddev = 1.0;
constexpr float tolerance = 1e-5f;
constexpr int distr_rank_single = 0;

} // anonymous namespace

template<typename T>
void check_randn_vs_tensor_api(const std::vector<Index>& shape)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());
    std::vector<Index> start(shape.size(), 0);

    // --- TensorGraph path ---
    TensorGraph graph("randn_test");
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    dst_node->mark_output(true);

    gt::randn(dst_node, start, shape, seed, mean, stddev);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path ---
    nntile::tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, distr_rank_single);
    nntile::tensor::Tensor<T> dst_t(traits, distr);

    nntile::tensor::randn<T>(dst_t, start, shape, seed, mean, stddev);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst_t.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tolerance);
    }
}

TEST_CASE("TensorGraph randn structure", "[graph][tensor]")
{
    TensorGraph graph("test");

    auto* dst = graph.data({4, 5}, "dst");
    gt::randn(dst, {0, 0}, {4, 5}, seed, mean, stddev);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "RANDN");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE("TensorGraph randn rejects null", "[graph][tensor]")
{
    TensorGraph graph("test");

    REQUIRE_THROWS_AS(
        gt::randn(nullptr, {0, 0}, {4, 5}, seed, mean, stddev),
        std::invalid_argument);
}

TEST_CASE("TensorGraph randn rejects mismatched start/underlying_shape",
    "[graph][tensor]")
{
    TensorGraph graph("test");
    auto* dst = graph.data({4, 5}, "dst");

    REQUIRE_THROWS_AS(
        gt::randn(dst, {0}, {4, 5}, seed, mean, stddev),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        gt::randn(dst, {0, 0}, {4}, seed, mean, stddev),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph randn matches nntile::tensor::randn", "[graph][tensor]")
{
    const auto shape = GENERATE(
        std::vector<Index>{4, 5},
        std::vector<Index>{6},
        std::vector<Index>{2, 3, 4});

    check_randn_vs_tensor_api<nntile::fp32_t>(shape);
}
