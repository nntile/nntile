/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/module/module.cc
 * Tests for Module base class.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"
#include "nntile/module/module.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::module;

namespace
{

class DummyModule final : public Module
{
public:
    DummyModule(NNGraph& graph, const std::string& name)
        : Module(graph, name)
    {
    }

    NNGraph::TensorNode& build_forward(NNGraph::TensorNode& input) override
    {
        input_tensor_ = &input;
        std::vector<Index> output_shape = input.shape();
        output_tensor_ = &graph_.tensor(
            output_shape,
            tensor_name("output"),
            input.dtype(),
            graph_.requires_grad(input));
        return *output_tensor_;
    }

    void build_backward() override
    {
        if(!input_tensor_ || !output_tensor_)
        {
            throw std::runtime_error(
                "DummyModule::build_backward: forward not built");
        }
        if(output_tensor_->grad() == nullptr)
        {
            throw std::runtime_error(
                "DummyModule::build_backward: missing output grad");
        }
    }
};

} // namespace

TEST_CASE("Module RegisterParameterAndBuffer", "[module]")
{
    NNGraph g("module");
    DummyModule module(g, "mod");

    auto& param = g.tensor({2, 3}, "param", DataType::FP32, true);
    auto& buffer = g.tensor({4}, "buffer", DataType::FP32, true);

    module.register_parameter("weight", &param);
    module.register_buffer("running", &buffer);

    REQUIRE(module.parameters().size() == 1);
    REQUIRE(module.named_parameters().size() == 1);
    REQUIRE(module.parameters()[0] == &param);
    REQUIRE(module.named_parameters()[0].first == "weight");
    REQUIRE(module.named_parameters()[0].second == &param);

    REQUIRE(module.buffers().size() == 1);
    REQUIRE(module.named_buffers().size() == 1);
    REQUIRE(module.buffers()[0] == &buffer);
    REQUIRE(module.named_buffers()[0].first == "running");
    REQUIRE(module.named_buffers()[0].second == &buffer);

    REQUIRE_FALSE(g.requires_grad(buffer));
    REQUIRE(module.tensor_name("x") == "mod_x");
    REQUIRE(module.grad_name("x") == "mod_x_grad");
}

TEST_CASE("Module RegisterNullPointers", "[module]")
{
    NNGraph g("module");
    DummyModule module(g, "mod");

    REQUIRE_THROWS_AS(
        module.register_parameter("weight", nullptr),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        module.register_buffer("running", nullptr),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        module.register_module("child", nullptr),
        std::invalid_argument);
}

TEST_CASE("Module RecursiveParametersAndModules", "[module]")
{
    NNGraph g("module");
    DummyModule parent(g, "parent");
    DummyModule child(g, "child");

    parent.register_module("child", &child);

    auto& parent_param = g.tensor({2, 2}, "parent_param", DataType::FP32);
    auto& child_param = g.tensor({3, 4}, "child_param", DataType::FP32);
    parent.register_parameter("p", &parent_param);
    child.register_parameter("w", &child_param);

    auto params = parent.parameters_recursive();
    REQUIRE(params.size() == 2);
    REQUIRE(std::find(params.begin(), params.end(), &parent_param) !=
        params.end());
    REQUIRE(std::find(params.begin(), params.end(), &child_param) !=
        params.end());

    auto named_params = parent.named_parameters_recursive();
    REQUIRE(named_params.size() == 2);
    REQUIRE(std::any_of(
        named_params.begin(), named_params.end(),
        [&](const auto& entry)
        {
            return entry.first == "parent.p" && entry.second == &parent_param;
        }));
    REQUIRE(std::any_of(
        named_params.begin(), named_params.end(),
        [&](const auto& entry)
        {
            return entry.first == "parent.child.w" &&
                entry.second == &child_param;
        }));

    auto children = parent.children();
    REQUIRE(children.size() == 1);
    REQUIRE(children[0] == &child);

    auto modules = parent.modules();
    REQUIRE(modules.size() == 2);
    REQUIRE(modules[0] == &parent);
    REQUIRE(modules[1] == &child);

    g.get_or_create_grad(parent_param, "parent_param_grad");
    g.get_or_create_grad(child_param, "child_param_grad");
    auto grads = parent.parameter_gradients_recursive();
    REQUIRE(grads.size() == 2);

    auto text = parent.to_string();
    REQUIRE(text.find("parent()") != std::string::npos);
    REQUIRE(text.find("(p): Parameter(") != std::string::npos);
    REQUIRE(text.find("(child):") != std::string::npos);
}
