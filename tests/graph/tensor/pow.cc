/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/tensor/pow.cc
 * Test TensorGraph pow operation against nntile::tensor::pow.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include <numeric>

#include "context_fixture.hh"
#include "nntile/graph/tensor/pow.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/pow.hh"
#include "nntile/tensor/tensor.hh"

using namespace nntile;
using namespace nntile::graph;

namespace
{

constexpr Scalar alpha = 1.0;
constexpr Scalar exponent = 2.0;

} // anonymous namespace

template<typename T>
void check_pow_vs_tensor_api(
    const std::vector<Index>& shape,
    Scalar alpha_val,
    Scalar exponent_val)
{
    using Y = typename T::repr_t;
    const Index nelems = std::accumulate(
        shape.begin(), shape.end(), Index(1), std::multiplies<>());

    // --- TensorGraph path ---
    TensorGraph graph("pow_test");
    auto* dst_node = graph.data(shape, "dst", DataType::FP32);
    dst_node->mark_input(true);
    dst_node->mark_output(true);

    pow(alpha_val, exponent_val, dst_node);

    TensorGraph::Runtime runtime(graph);
    runtime.compile();

    // Use positive values for pow (avoid complex/NaN for negative base)
    std::vector<float> dst_data(nelems);
    for(Index i = 0; i < nelems; ++i)
    {
        dst_data[i] = static_cast<float>(Y(i + 1));
    }

    runtime.bind_data("dst", dst_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> graph_result = runtime.get_output<float>("dst");

    // --- Direct tensor API path (same input data) ---
    tensor::TensorTraits traits(shape, shape);
    std::vector<int> distr(traits.grid.nelems, 0);
    tensor::Tensor<T> dst(traits, distr);

    {
        auto tile = dst.get_tile(0);
        auto loc = tile.acquire(STARPU_W);
        for(Index i = 0; i < nelems; ++i)
        {
            loc[i] = static_cast<Y>(dst_data[i]);
        }
        loc.release();
    }

    tensor::pow<T>(alpha_val, exponent_val, dst);
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

TEST_CASE("TensorGraph pow structure", "[graph][tensor]")
{
    constexpr Index dim0 = 4;
    constexpr Index dim1 = 5;

    TensorGraph graph("test");

    auto* dst = graph.data({dim0, dim1}, "dst");

    pow(alpha, exponent, dst);

    REQUIRE(graph.num_data() == 1);
    REQUIRE(graph.num_ops() == 1);

    const auto& ops = graph.ops();
    REQUIRE(ops[0]->op_name() == "POW");
    REQUIRE(ops[0]->inputs().size() == 1);
    REQUIRE(ops[0]->outputs().size() == 1);
    REQUIRE(ops[0]->outputs()[0] == dst);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "TensorGraph pow matches tensor::pow", "[graph][tensor]")
{
    const auto [alpha_val, exponent_val, shape] = GENERATE(
        std::tuple{1.0, 2.0, std::vector<Index>{4, 5}},
        std::tuple{2.0, 0.5, std::vector<Index>{6}},
        std::tuple{1.0, 3.0, std::vector<Index>{2, 3}},
        std::tuple{0.5, 2.0, std::vector<Index>{1, 10}});

    check_pow_vs_tensor_api<nntile::fp32_t>(shape, alpha_val, exponent_val);
}
