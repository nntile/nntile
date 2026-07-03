/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/tests/nn_graph/hypot.cc
 * Test NNGraph hypot autograd operation.
 *
 * @version 1.1.0
 * */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
namespace gt = nntile::tensor;

namespace
{

constexpr Index dim_2 = 2;
constexpr Index dim_3 = 3;

} // anonymous namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph hypot structure",
    "[graph][nn_graph]")
{
    const auto [alpha, beta] = GENERATE(std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    NNGraph g("hypot_structure");
    auto *x = g.tensor({dim_3, dim_2}, DataType::FP32)->set_name("x");
    auto *y = g.tensor({dim_3, dim_2}, DataType::FP32)->set_name("y");
    auto *z = hypot(x, y, alpha, beta);

    REQUIRE(z != nullptr);
    REQUIRE(z->has_producer());
    REQUIRE(z->shape() == (std::vector<Index>{dim_3, dim_2}));
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops()[0]->op_name() == "HYPOT");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph hypot forward",
    "[graph][nn_graph]")
{
    const auto [alpha, beta] = GENERATE(std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    NNGraph g("hypot_forward");
    auto *x = g.tensor({6, 7}, DataType::FP32, false)->set_name("x");
    auto *y = g.tensor({6, 7}, DataType::FP32, false)->set_name("y");
    auto *z = hypot(x, y, alpha, beta);

    x->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    x->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});

    x->mark_input(true);
    y->mark_input(true);
    z->mark_output(true);

    std::vector<float> x_data(6 * 7);
    std::vector<float> y_data(6 * 7);
    for (Index i = 0; i < 6 * 7; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i + 1);
        y_data[i] = 0.2f * static_cast<float>(-i - 1);
    }

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(y, y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> out = runtime.get_output<float>(z);
    REQUIRE(out.size() == 6 * 7);
    for (float value : out)
    {
        REQUIRE(value >= 0.0f);
    }
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph hypot backward throws",
    "[graph][nn_graph]")
{
    const auto [alpha, beta, grad_fill_val] =
        GENERATE(std::tuple{Scalar(1.0), Scalar(1.0), Scalar(1.0)},
            std::tuple{Scalar(2.0), Scalar(0.5), Scalar(-1.0)},
            std::tuple{Scalar(0.5), Scalar(-1.0), Scalar(0.5)});

    NNGraph g("hypot_backward_throws");
    auto *x = g.tensor({dim_3, dim_2}, DataType::FP32, true)->set_name("x");
    auto *y = g.tensor({dim_3, dim_2}, DataType::FP32, true)->set_name("y");
    auto *z = hypot(x, y, alpha, beta);

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());

    REQUIRE_THROWS_AS(z->backward(), std::runtime_error);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::compare_float_vectors;
using nntile::test::nn_pytorch_tile_heterogeneous_rank2_6x7;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph hypot forward matches PyTorch",
    "[graph][nn_graph][pytorch]")
{
    const auto [alpha, beta] = GENERATE(std::tuple{Scalar(1.0), Scalar(1.0)},
        std::tuple{Scalar(2.0), Scalar(0.5)},
        std::tuple{Scalar(0.5), Scalar(-1.0)});

    constexpr Index dim0 = 6;
    constexpr Index dim1 = 7;
    constexpr Index nelems = dim0 * dim1;

    std::vector<float> x_data(nelems);
    std::vector<float> y_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i + 1);
        y_data[i] = 0.2f * static_cast<float>(-i - 1);
    }

    NNGraph g("hypot_pytorch");
    auto *x = g.tensor({dim0, dim1}, DataType::FP32, false)->set_name("x");
    auto *y = g.tensor({dim0, dim1}, DataType::FP32, false)->set_name("y");
    auto *z = hypot(x, y, alpha, beta);

    nn_pytorch_tile_heterogeneous_rank2_6x7(x);

    x->mark_input(true);
    y->mark_input(true);
    z->mark_output(true);

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(x, x_data);
    runtime.bind_data(y, y_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out = runtime.get_output<float>(z);

    auto x_pt = torch::from_blob(x_data.data(),
        {dim0, dim1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto y_pt = torch::from_blob(y_data.data(),
        {dim0, dim1},
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);

    auto z_pt = torch::hypot(alpha * x_pt, beta * y_pt).contiguous();
    compare_float_vectors(nntile_out, z_pt);
}

#endif
