/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/module/activation.cc
 * Tests for Activation module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/functional/activation.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/graph/module/activation.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "nntile/graph/tensor/graph.hh"
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("Activation AllTypes", "[module]")
{
    NNGraph g("activation");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);

    Activation gelu(&g, "gelu", ActivationType::GELU);
    Activation gelutanh(&g, "gelutanh", ActivationType::GELUTANH);
    Activation relu(&g, "relu", ActivationType::RELU);
    Activation silu(&g, "silu", ActivationType::SILU);

    auto* out_gelu = gelu.forward(input);
    REQUIRE(out_gelu->shape() == std::vector<Index>({2, 3}));

    REQUIRE(gelu.type() == ActivationType::GELU);
    REQUIRE(gelutanh.type() == ActivationType::GELUTANH);
    REQUIRE(relu.type() == ActivationType::RELU);
    REQUIRE(silu.type() == ActivationType::SILU);
}

TEST_CASE("Activation TypeFromString", "[module]")
{
    REQUIRE(activation_type_from_string("gelu") == ActivationType::GELU);
    REQUIRE(activation_type_from_string("gelutanh") == ActivationType::GELUTANH);
    REQUIRE(activation_type_from_string("relu") == ActivationType::RELU);
    REQUIRE(activation_type_from_string("silu") == ActivationType::SILU);

    REQUIRE_THROWS_AS(activation_type_from_string("unknown"),
                      std::invalid_argument);
}

TEST_CASE("Activation BackwardCreatesGradients", "[module]")
{
    NNGraph g("activation_bwd");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32, true);
    Activation activation(&g, "activation", ActivationType::GELU);
    auto* output = activation.forward(input);

    g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(1.0), output->grad()->data());
    output->backward();

    REQUIRE(input->grad() != nullptr);
    REQUIRE(input->grad()->shape() == std::vector<Index>({2, 3}));
}

#ifdef NNTILE_HAVE_TORCH

namespace
{

torch::Tensor apply_activation_pt(torch::Tensor x, ActivationType t)
{
    switch(t)
    {
        case ActivationType::RELU:
            return torch::nn::functional::relu(x);
        case ActivationType::GELU:
            return torch::nn::functional::gelu(x);
        case ActivationType::SILU:
            return torch::nn::functional::silu(x);
        case ActivationType::GELUTANH:
            return torch::nn::functional::gelu(x,
                torch::nn::functional::GELUFuncOptions().approximate("tanh"));
        default:
            throw std::invalid_argument("Unsupported activation for test");
    }
}

} // anonymous namespace

using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Activation forward and backward match PyTorch", "[module][pytorch]")
{
    const auto [shape, act_type, grad_fill_val] = GENERATE(
        std::tuple{std::vector<Index>{2, 3}, ActivationType::RELU, Scalar(1.0)},
        std::tuple{std::vector<Index>{4, 6}, ActivationType::GELU, Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 3, 4}, ActivationType::SILU, Scalar(1.0)},
        std::tuple{std::vector<Index>{3, 5}, ActivationType::GELUTANH, Scalar(1.0)},
        std::tuple{std::vector<Index>{1, 8}, ActivationType::RELU, Scalar(-1.0)},
        std::tuple{std::vector<Index>{6}, ActivationType::GELU, Scalar(1.0)},
        std::tuple{std::vector<Index>{2, 4}, ActivationType::SILU, Scalar(-1.0)});

    Index nelems = 1;
    for(auto s : shape)
        nelems *= s;

    std::vector<float> input_data(nelems);
    for(Index i = 0; i < nelems; ++i)
        input_data[i] = 0.1f * static_cast<float>(i - nelems / 2);

    NNGraph g("activation_pytorch");
    auto* input = g.tensor(shape, "input", DataType::FP32, true);
    Activation activation(&g, "activation", act_type);
    auto* output = activation.forward(input);

    input->mark_input(true);
    output->mark_output(true);

    auto [grad_output_tensor, _] = g.get_or_create_grad(output, "output_grad");
    gt::fill(Scalar(grad_fill_val), grad_output_tensor->data());
    output->backward();

    input->grad()->mark_output(true);

    TileGraph runtime_tile = TileGraph::from_tensor_graph(g.tensor_graph());


    TileGraph::Runtime runtime(runtime_tile);
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out =
        runtime.get_output<float>(output->name());
    std::vector<float> nntile_grad_input =
        runtime.get_output<float>(input->grad()->name());

    std::vector<::int64_t> shape_pt(shape.begin(), shape.end());
    auto input_pt = torch::from_blob(input_data.data(), shape_pt,
        torch::TensorOptions().dtype(torch::kFloat32)).clone()
            .set_requires_grad(true);
    auto out_pt = apply_activation_pt(input_pt, act_type);
    auto grad_output = torch::full(shape_pt, static_cast<float>(grad_fill_val),
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    out_pt.backward(grad_output);

    REQUIRE(nntile_out.size() == static_cast<size_t>(nelems));
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - out_pt.data_ptr<float>()[i]) <
            pytorch_tolerance);

    compare_float_vectors(nntile_grad_input, input_pt.grad());
}

#endif // NNTILE_HAVE_TORCH
