/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/test_compiled_ops.cc
 * Tests for compiled graph operations.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "nntile/context.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;

// Fixture to initialize NNTile context for graph tests
class GraphTestFixture
{
protected:
    nntile::Context context;
public:
    GraphTestFixture():
        context(
            1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0
        )
    {}
};

namespace
{

float gelu_backward_expected(float x, float dy, float dx)
{
    const double pi = 3.141592653589793238462643383279502884;
    const double f1 = -1.0 / std::sqrt(2.0);
    const double f2 = 1.0 / std::sqrt(2.0 * pi);
    const double xd = static_cast<double>(x);
    const double dyd = static_cast<double>(dy);
    const double dxd = static_cast<double>(dx);
    const double exp_x = std::exp(-0.5 * xd * xd);
    const double erfc_x = std::erfc(f1 * xd);
    const double grad = (xd * f2 * exp_x + 0.5 * erfc_x) * dyd;
    return static_cast<float>(dxd + grad);
}

} // namespace

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GemmAccumulation",
    "[graph]")
{
    LogicalGraph g("test");

    auto& a = g.tensor({2, 3}, "a", DataType::FP32);
    auto& b = g.tensor({3, 4}, "b", DataType::FP32);
    auto& c = g.tensor({2, 4}, "c", DataType::FP32);

    gemm(a, b, c, 2.0f, 0.5f);

    auto compiled = CompiledGraph::compile(g);

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    };
    std::vector<float> c_data(8, 1.0f);

    compiled.bind_data("a", a_data);
    compiled.bind_data("b", b_data);
    compiled.bind_data("c", c_data);

    compiled.execute();
    compiled.wait();

    auto out = compiled.get_output<float>("c");

    const std::vector<float> expected = {
        44.5f, 56.5f, 98.5f, 128.5f, 152.5f, 200.5f, 206.5f, 272.5f
    };
    REQUIRE(out.size() == expected.size());
    for(size_t i = 0; i < out.size(); ++i)
    {
        REQUIRE(out[i] == Catch::Approx(expected[i]));
    }
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ClearBool",
    "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({4}, "x", DataType::BOOL);
    clear(x);

    auto compiled = CompiledGraph::compile(g);

    compiled.bind_data("x", std::vector<float>{1.0f, 0.0f, 2.0f, -3.0f});
    compiled.execute();
    compiled.wait();

    auto out = compiled.get_output<float>("x");
    REQUIRE(out.size() == 4);
    for(float v : out)
    {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluBackward",
    "[graph]")
{
    LogicalGraph g("test");

    auto& x = g.tensor({3}, "x", DataType::FP32);
    auto& dy = g.tensor({3}, "dy", DataType::FP32);
    auto& dx = g.tensor({3}, "dx", DataType::FP32);

    gelu_backward(x, dy, dx);

    auto compiled = CompiledGraph::compile(g);

    std::vector<float> x_data = {-1.0f, 0.0f, 1.0f};
    std::vector<float> dy_data = {0.1f, 1.0f, -0.5f};
    std::vector<float> dx_data = {0.2f, -0.1f, 0.0f};

    compiled.bind_data("x", x_data);
    compiled.bind_data("dy", dy_data);
    compiled.bind_data("dx", dx_data);

    compiled.execute();
    compiled.wait();

    auto out = compiled.get_output<float>("dx");
    REQUIRE(out.size() == x_data.size());

    for(size_t i = 0; i < out.size(); ++i)
    {
        float expected = gelu_backward_expected(
            x_data[i], dy_data[i], dx_data[i]);
        REQUIRE(out[i] == Catch::Approx(expected).epsilon(1e-3f));
    }
}
