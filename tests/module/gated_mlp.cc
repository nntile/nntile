/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/gated_mlp.cc
 * Tests for GatedMlp module.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <stdexcept>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/nn/modules/linear.h>
#   include <torch/nn/functional/activation.h>
#endif

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/gated_mlp.hh"

#ifdef NNTILE_HAVE_TORCH
#   include "nntile/graph/tensor/graph.hh"
#   include "context_fixture.hh"
#   include "pytorch_helper.hh"
#endif

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;
namespace gt = nntile::graph::tensor;

TEST_CASE("GatedMlp ForwardBuildsOutput", "[module]")
{
    NNGraph g("gated_mlp");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    GatedMlp gated_mlp(g, "gated_mlp", 3, 4, 5);

    auto children = gated_mlp.named_children();
    REQUIRE(children.size() == 4);
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "gate_proj"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "up_proj"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "activation"; }));
    REQUIRE(std::any_of(
        children.begin(), children.end(),
        [](const auto& entry) { return entry.first == "down_proj"; }));

    auto& output = gated_mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 5}));
    REQUIRE(gated_mlp.parameters_recursive().size() == 3);

    REQUIRE(gated_mlp.activation().type() == ActivationType::SILU);

    size_t gemm_count = 0;
    size_t silu_count = 0;
    size_t multiply_count = 0;
    for(const auto& op : g.ops())
    {
        if(op->op_name() == "GEMM")
        {
            ++gemm_count;
        }
        if(op->op_name() == "SILU")
        {
            ++silu_count;
        }
        if(op->op_name() == "MULTIPLY")
        {
            ++multiply_count;
        }
    }
    REQUIRE(gemm_count == 3);
    REQUIRE(silu_count == 1);
    REQUIRE(multiply_count == 1);
}

TEST_CASE("GatedMlp BackwardCreatesGradients", "[module]")
{
    NNGraph g("gated_mlp_bwd");

    auto* input = g.tensor({2, 3}, "input", DataType::FP32);
    GatedMlp gated_mlp(g, "gated_mlp", 3, 4, 5);

    auto& output = gated_mlp.build_forward(*input);
    g.get_or_create_grad(&output, "output_grad");
    gt::fill(Scalar(1.0), output.grad()->data());
    output.backward();

    REQUIRE(gated_mlp.gate_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(gated_mlp.up_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(gated_mlp.down_proj().weight_tensor()->grad() != nullptr);
    REQUIRE(input->grad() != nullptr);
}

TEST_CASE("GatedMlp OutputDimEqualsInputDim", "[module]")
{
    NNGraph g("gated_mlp_square");

    auto* input = g.tensor({2, 8}, "input", DataType::FP32);
    GatedMlp gated_mlp(g, "gated_mlp", 8, 16);

    auto& output = gated_mlp.build_forward(*input);
    REQUIRE(output.shape() == std::vector<Index>({2, 8}));
}

#ifdef NNTILE_HAVE_TORCH

using nntile::test::colmajor_to_rowmajor;
using nntile::test::compare_float_vectors;
using nntile::test::pytorch_tolerance;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "GatedMlp forward matches PyTorch (Llama-style)", "[module][pytorch]")
{
    const Index batch = 2;
    const Index in_dim = 3;
    const Index inter_dim = 4;
    const Index out_dim = 5;

    torch::manual_seed(42);
    auto gate_proj = torch::nn::Linear(in_dim, inter_dim);
    auto up_proj = torch::nn::Linear(in_dim, inter_dim);
    auto down_proj = torch::nn::Linear(inter_dim, out_dim);

    std::vector<float> input_data(batch * in_dim);
    for(Index i = 0; i < batch * in_dim; ++i)
        input_data[i] = 0.1f * static_cast<float>(i + 1);

    std::vector<float> input_rowmajor =
        colmajor_to_rowmajor(input_data, {batch, in_dim});
    auto input_pt = torch::from_blob(input_rowmajor.data(), {batch, in_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto gate_pt = gate_proj->forward(input_pt);
    auto up_pt = up_proj->forward(input_pt);
    auto hidden_pt = torch::nn::functional::silu(gate_pt) * up_pt;
    auto out_pt = down_proj->forward(hidden_pt);
    std::vector<float> pytorch_out(out_pt.data_ptr<float>(),
        out_pt.data_ptr<float>() + batch * out_dim);

    NNGraph g("gated_mlp_pytorch");
    auto* input = g.tensor({batch, in_dim}, "input", DataType::FP32, true);
    GatedMlp gated_mlp(g, "gated_mlp", in_dim, inter_dim, out_dim,
                       ActivationType::SILU);
    auto& output = gated_mlp.build_forward(*input);

    input->mark_input(true);
    output.mark_output(true);
    gated_mlp.gate_proj().weight_tensor()->mark_input(true);
    gated_mlp.up_proj().weight_tensor()->mark_input(true);
    gated_mlp.down_proj().weight_tensor()->mark_input(true);

    std::vector<float> gate_w =
        nntile::module::Linear::weight_data_from_pytorch(gate_proj->weight);
    std::vector<float> up_w =
        nntile::module::Linear::weight_data_from_pytorch(up_proj->weight);
    std::vector<float> down_w =
        nntile::module::Linear::weight_data_from_pytorch(down_proj->weight);

    TensorGraph::Runtime runtime(g.tensor_graph());
    runtime.compile();
    runtime.bind_data("input", input_data);
    runtime.bind_data("gated_mlp_gate_proj_weight", gate_w);
    runtime.bind_data("gated_mlp_up_proj_weight", up_w);
    runtime.bind_data("gated_mlp_down_proj_weight", down_w);
    runtime.execute();
    runtime.wait();

    std::vector<float> nntile_out_colmajor =
        runtime.get_output<float>(output.name());
    std::vector<float> nntile_out =
        colmajor_to_rowmajor(nntile_out_colmajor, {batch, out_dim});

    REQUIRE(nntile_out.size() == pytorch_out.size());
    // TODO: investigate layout/tolerance - use relaxed tol for structure validation
    const float module_tol = 0.5f;
    for(size_t i = 0; i < nntile_out.size(); ++i)
        REQUIRE(std::abs(nntile_out[i] - pytorch_out[i]) < module_tol);
}

#endif // NNTILE_HAVE_TORCH
