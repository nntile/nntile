/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/tile/gpt2_block_schedule.cc
 * GPT-2 MLP: tiling.json axes, execution.json write/load, execute.
 *
 * @version 1.1.0
 * */

#include "context_fixture.hh"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <nntile/context.hh>
#include <nntile/core/execution_schedule.hh>
#include <nntile/model/gpt2/gpt2_mlp.hh>
#include <nntile/tensor/ops/fill.hh>
#include <random>
#include <vector>

#include "gpt2_axis_naming.hh"
#include "tiling_config_json.hh"

using namespace nntile;
using namespace nntile::model::gpt2;
using namespace nntile::examples;
namespace gt = nntile::tensor;

namespace
{

Gpt2Config make_tiny_block_config()
{
    Gpt2Config c;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 1;
    c.num_attention_heads = 2;
    c.max_position_embeddings = 128;
    c.layer_norm_eps = 1e-5f;
    c.validate();
    return c;
}

void init_random_parameter_hints(module::Module &mod, unsigned seed)
{
    std::mt19937 gen(seed);
    for (NNGraph::TensorNode *tensor : mod.parameters_recursive())
    {
        Index nelems = 1;
        for (Index d : tensor->shape())
        {
            nelems *= d;
        }
        float const limit = 0.05f;
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::vector<float> data(static_cast<size_t>(nelems));
        for (float &v : data)
        {
            v = dist(gen);
        }
        std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        tensor->data()->set_bind_hint(std::move(bytes));
    }
    mod.mark_parameters_input_recursive();
}

void bind_parameters_runtime(Runtime &rt, module::Module &mod, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (NNGraph::TensorNode *tensor : mod.parameters_recursive())
    {
        Index nelems = 1;
        for (Index d : tensor->shape())
        {
            nelems *= d;
        }
        std::vector<float> data(static_cast<size_t>(nelems));
        for (float &v : data)
        {
            v = dist(gen);
        }
        rt.bind_data(tensor, data);
    }
}

} // namespace

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "Gpt2MLP tiled forward with execution.json round-trip",
    "[gpt2][schedule]")
{
    Gpt2Config const cfg = make_tiny_block_config();
    constexpr Index n_seq = 4;
    constexpr Index n_batch = 2;

    NNGraph graph("gpt2_mlp_schedule");
    Gpt2MLP mlp(&graph, "mlp", cfg);
    auto *input = graph.tensor({cfg.hidden_size, n_seq, n_batch}, DataType::FP32)
                      ->set_name("input");
    input->mark_input(true);
    auto *output = mlp.forward(input);
    REQUIRE(output != nullptr);
    output->mark_output(true);

    init_random_parameter_hints(mlp, 11u);

    FlatTilingSpec spec;
    spec.defaults["seq_len"] = {2, 2};
    spec.defaults["batch_size"] = {1, 1};
    name_gpt2_training_axis_groups(graph.tensor_graph(), cfg, n_seq, n_batch);
    apply_flat_tiling_spec(graph.tensor_graph(), spec);

    auto [grad_out, _] = graph.get_or_create_grad(output, "dloss");
    gt::fill(Scalar(1.0f), grad_out->data());
    output->backward(true);

    graph.finish_phase();
    graph.lower_and_compile();
    Runtime &runtime = graph.runtime();

    char const *const exec_path = "/tmp/nntile_gpt2_mlp_execution.json";
    write_execution_schedule_json(
        runtime.generate_round_robin_execution_schedule(), exec_path);
    runtime.load_execution_schedule(exec_path);

    std::vector<float> in_data(
        static_cast<size_t>(cfg.hidden_size * n_seq * n_batch), 0.1f);
    runtime.bind_data(input, in_data);
    bind_parameters_runtime(runtime, mlp, 11u);

    runtime.execute();
    runtime.wait();

    std::vector<float> out =
        runtime.get_output<float>(output);
    REQUIRE(!out.empty());
    float sum = 0.f;
    for (float v : out)
    {
        sum += std::abs(v);
    }
    REQUIRE(sum > 0.f);
}
