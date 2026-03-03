/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/hypot.cc
 * Test TensorGraph hypot operation against nntile::tensor::hypot.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/hypot.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/hypot.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Scalar alpha = 1.0;
constexpr Scalar beta = 1.0;

} // anonymous namespace

template<typename T>
void check_hypot_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha,
    Scalar beta)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("hypot_test");
    auto* src1_node = graph.data(shape, "src1", DataType::FP32);
    auto* src2_node = graph.data(shape, "src2", DataType::FP32);
    src1_node->mark_input(true);
    src2_node->mark_input(true);

    auto* dst_node = hypot(alpha, src1_node, beta, src2_node, "dst");
    dst_node->mark_output(true);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    std::vector<float> src1_data(nelems), src2_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        src1_data[i] = static_cast<float>(Y(i + 1));
        src2_data[i] = static_cast<float>(Y(-i - 1));
    }

    runtime.bind_data("src1", src1_data);
    runtime.bind_data("src2", src2_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    tensor::Tensor<T> src1(traits, distr);
    tensor::Tensor<T> src2(traits, distr);
    tensor::Tensor<T> dst(traits, distr);

    {
        auto tile1 = src1.get_tile(0);
        auto tile2 = src2.get_tile(0);
        auto loc1 = tile1.acquire(STARPU_W);
        auto loc2 = tile2.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc1[i] = static_cast<Y>(src1_data[i]);
            loc2[i] = static_cast<Y>(src2_data[i]);
        }
        loc1.release();
        loc2.release();
    }

    tensor::hypot<T>(alpha, src1, beta, src2, dst);
    starpu_task_wait_for_all();

    std::vector<float> tensor_result(nelems);
    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_R);
        for(Index i = 0; i < nelems; ++i)
        {
            tensor_result[i] = static_cast<float>(loc[i]);
        }
        loc.release();
    }

    constexpr float tol = 1e-5f;
    REQUIRE(graph_result.size() == tensor_result.size());
    for(size_t i = 0; i < graph_result.size(); ++i)
    {
        REQUIRE(std::abs(graph_result[i] - tensor_result[i]) < tol);
    }
}

TEST_CASE("TensorGraph hypot structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* src1 = graph.data({dim0, dim1}, "src1");
    auto* src2 = graph.data({dim0, dim1}, "src2");

    auto* dst = hypot(alpha, src1, beta, src2, "dst");

    REQUIRE(graph.num_data() == 3);
    REQUIRE(graph.num_ops() == 1);
    REQUIRE(dst->shape()[0] == dim0);
    REQUIRE(dst->shape()[1] == dim1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "HYPOT");
    REQUIRE(ops[0]->inputs().size() == 2);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph hypot matches tensor::hypot", "[graph][tensor]")
{
    const auto [alpha, beta, shape] = GENERATE(
        std::tuple{1.0, 1.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 3.0, std::vector<Index>{4, 5}},
        std::tuple{0.5, -1.0, std::vector<Index>{6}},
        std::tuple{1.0, 2.0, std::vector<Index>{3, 4}});

    check_hypot_vs_tensor_api<nntile::fp32_t>(shape, alpha, beta);
}
