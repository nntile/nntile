/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/layer_norm.cc
 * Tests for LayerNorm module.
 *
 * @version 1.1.0
 * */

#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#include <torch/nn/modules/normalization.h>
#endif

#include "nntile/graph.hh"
#include "nntile/graph/module/layer_norm.hh"
#include "nntile/graph/tensor/graph.hh"

#ifdef NNTILE_HAVE_TORCH
#include "context_fixture.hh"
#include "pytorch_helper.hh"
#include "pytorch_tile_helpers.hh"
#endif

using namespace nntile::core;
using namespace nntile::graph;
using namespace nntile::graph::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("LayerNorm ConstructorCreatesParameters", "[module]")
{
    NNGraph g("layer_norm");

    LayerNorm ln(&g, "ln", 64, 0, 1e-5f);
    REQUIRE(ln.gamma_tensor() != nullptr);
    REQUIRE(ln.beta_tensor() != nullptr);
    REQUIRE(ln.gamma_tensor()->shape() == std::vector<Index>({64}));
    REQUIRE(ln.beta_tensor()->shape() == std::vector<Index>({64}));
    REQUIRE(ln.gamma_tensor()->name() == "ln_gamma");
    REQUIRE(ln.beta_tensor()->name() == "ln_beta");
    REQUIRE(ln.parameters().size() == 2);
}

TEST_CASE("LayerNorm Callable", "[module]")
{
    NNGraph g("layer_norm_callable");
    auto *input = g.tensor({4, 64}, DataType::FP32)->set_name("input");
    LayerNorm ln(&g, "ln", 64, 1, 1e-5f);
    auto *output = ln.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({4, 64}));
}

TEST_CASE("LayerNorm BuildForward", "[module]")
{
    NNGraph g("layer_norm");

    auto *input = g.tensor({2, 3, 4}, DataType::FP32)->set_name("input");
    LayerNorm ln(&g, "ln", 4, 2, 1e-5f);

    auto *output = ln.forward(input);
    REQUIRE(output->shape() == std::vector<Index>({2, 3, 4}));
    REQUIRE(output->name() == "ln_out");
    REQUIRE(g.num_ops() >= 1);
}

TEST_CASE("LayerNorm Repr", "[module]")
{
    NNGraph g("layer_norm");
    LayerNorm ln(&g, "ln", 768, 0, 1e-5f);
    std::string r = ln.repr();
    REQUIRE(r.find("LayerNorm") != std::string::npos);
    REQUIRE(r.find("768") != std::string::npos);
}

#ifdef NNTILE_HAVE_TORCH

using nntile::core::test::colmajor_to_rowmajor;
using nntile::core::test::compare_float_vectors;
using nntile::core::test::module_tile_all_untiled_axis_groups_heterogeneous;
using nntile::core::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "LayerNorm forward matches PyTorch",
    "[module][pytorch]")
{
    const auto [shape, axis] =
        GENERATE(std::tuple{std::vector<Index>{4, 64}, Index(1)},
            std::tuple{std::vector<Index>{6, 7}, Index(1)});

    Index nelems = 1;
    for (auto s : shape)
    {
        nelems *= s;
    }
    const Index normalized = shape[axis];

    std::vector<float> x_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    }
    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, shape);

    torch::manual_seed(7);
    auto ln_pt = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({static_cast<long>(normalized)}).eps(
            1e-5));

    std::vector<float> gamma_data(normalized);
    std::vector<float> beta_data(normalized);
    auto w = ln_pt->weight.accessor<float, 1>();
    auto b = ln_pt->bias.accessor<float, 1>();
    for (Index i = 0; i < normalized; ++i)
    {
        gamma_data[i] = w[static_cast<long>(i)];
        beta_data[i] = b[static_cast<long>(i)];
    }

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(),
        shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(false);
    auto y_pt = ln_pt->forward(x_pt);

    NNGraph g("layer_norm_fwd_pytorch");
    auto *input =
        g.tensor(shape, DataType::FP32, true)->set_name("input");
    LayerNorm ln(&g, "ln", normalized, axis, 1e-5f);
    auto *output = ln.forward(input);

    input->mark_input(true);
    ln.gamma_tensor()->mark_input(true);
    ln.beta_tensor()->mark_input(true);
    output->mark_output(true);

    module_tile_all_untiled_axis_groups_heterogeneous(g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(input, x_data);
    runtime.bind_data(ln.gamma_tensor(), gamma_data);
    runtime.bind_data(ln.beta_tensor(), beta_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor = runtime.get_output<float>(output);
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, shape);
    compare_float_vectors(nntile_out, y_pt);
}

TEST_CASE_METHOD(nntile::core::test::ContextFixture,
    "LayerNorm forward-backward matches PyTorch",
    "[module][pytorch]")
{
    const auto [shape, axis, grad_fill_val] =
        GENERATE(std::tuple{std::vector<Index>{3, 8}, Index(1), Scalar(1.0)},
            std::tuple{std::vector<Index>{6, 7}, Index(1), Scalar(-1.0)});

    Index nelems = 1;
    for (auto s : shape)
    {
        nelems *= s;
    }
    const Index normalized = shape[axis];

    std::vector<float> x_data(nelems);
    for (Index i = 0; i < nelems; ++i)
    {
        x_data[i] = 0.12f * static_cast<float>(i - nelems / 4);
    }
    std::vector<float> x_rowmajor = colmajor_to_rowmajor(x_data, shape);

    torch::manual_seed(11);
    auto ln_pt = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({static_cast<long>(normalized)}).eps(
            1e-5));

    std::vector<float> gamma_data(normalized);
    std::vector<float> beta_data(normalized);
    auto w = ln_pt->weight.accessor<float, 1>();
    auto b = ln_pt->bias.accessor<float, 1>();
    for (Index i = 0; i < normalized; ++i)
    {
        gamma_data[i] = w[static_cast<long>(i)];
        beta_data[i] = b[static_cast<long>(i)];
    }

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto x_pt = torch::from_blob(x_rowmajor.data(),
        shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32))
                    .clone()
                    .set_requires_grad(true);
    auto y_pt = ln_pt->forward(x_pt);
    auto grad_output = torch::full(shape_pt,
        static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    y_pt.backward(grad_output);

    NNGraph g("layer_norm_bwd_pytorch");
    auto *input =
        g.tensor(shape, DataType::FP32, true)->set_name("input");
    LayerNorm ln(&g, "ln", normalized, axis, 1e-5f);
    auto *output = ln.forward(input);

    input->mark_input(true);
    ln.gamma_tensor()->mark_input(true);
    ln.beta_tensor()->mark_input(true);

    auto [output_grad, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(grad_fill_val, output_grad->data());
    output->backward();

    ln.gamma_tensor()->grad()->mark_output(true);
    ln.beta_tensor()->grad()->mark_output(true);
    input->grad()->mark_output(true);

    module_tile_all_untiled_axis_groups_heterogeneous(g.tensor_graph());

    TileGraph tile_graph = TileGraph::from_tensor_graph(g.tensor_graph());
    Runtime runtime(tile_graph);
    runtime.compile();
    runtime.bind_data(input, x_data);
    runtime.bind_data(ln.gamma_tensor(), gamma_data);
    runtime.bind_data(ln.beta_tensor(), beta_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_grad_x_colmajor =
        runtime.get_output<float>(input->grad());
    std::vector<float> nntile_grad_x =
        colmajor_to_rowmajor(nntile_grad_x_colmajor, shape);
    compare_float_vectors(nntile_grad_x, x_pt.grad());
    compare_float_vectors(
        runtime.get_output<float>(ln.gamma_tensor()->grad()),
        ln_pt->weight.grad());
    compare_float_vectors(
        runtime.get_output<float>(ln.beta_tensor()->grad()),
        ln_pt->bias.grad());
}

#endif // NNTILE_HAVE_TORCH
