/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn_graph.cc
 * Tests for NNGraph class.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <string>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>

// Include other NNTile headers
#include "nntile/graph.hh"

using namespace nntile::graph;

TEST_CASE("NNGraph TensorNodeNullData", "[graph]")
{
    REQUIRE_THROWS_AS(NNGraph::TensorNode(nullptr), std::invalid_argument);
}

TEST_CASE("NNGraph TensorCreationAndLookup", "[graph]")
{
    NNGraph g("test");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32, false);

    REQUIRE(x.name() == "x");
    REQUIRE_FALSE(x.requires_grad());
    REQUIRE(g.get_tensor("x") == &x);
    REQUIRE(g.get_tensor("missing") == nullptr);
    REQUIRE(x.data_ptr() == g.logical_graph().get_tensor("x"));

    auto names = g.tensor_names();
    REQUIRE(names.size() == 1);
    REQUIRE(std::find(names.begin(), names.end(), "x") != names.end());
}

TEST_CASE("NNGraph AddOpNullInputsOutputs", "[graph]")
{
    NNGraph g("test");

    auto& x = g.tensor({2, 2}, "x", DataType::FP32);
    auto& y = g.tensor({2, 2}, "y", DataType::FP32);

    REQUIRE_THROWS_AS(
        g.add_op(OpType::GEMM, GemmAttrs{}, {&x, nullptr}, {&y}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        g.add_op(OpType::GELU, GeluAttrs{}, {&x}, {nullptr}),
        std::invalid_argument);

    g.add_op(OpType::GELU, GeluAttrs{}, {&x}, {&y});
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.logical_graph().ops().front()->type() == OpType::GELU);
}

TEST_CASE("NNGraph GradHelpersAndToString", "[graph]")
{
    NNGraph g("grad");

    auto& x = g.tensor({4}, "x", DataType::FP32, false);
    REQUIRE_FALSE(g.requires_grad(x));

    g.set_requires_grad(x, true);
    REQUIRE(g.requires_grad(x));
    g.set_requires_grad(x, false);
    REQUIRE_FALSE(g.requires_grad(x));

    auto& grad = g.get_or_create_grad(x, "x_grad");
    REQUIRE(x.has_grad());
    REQUIRE(&grad == x.grad());
    REQUIRE(x.requires_grad());
    REQUIRE_FALSE(grad.requires_grad());
    REQUIRE_FALSE(g.requires_grad(grad));
    REQUIRE(g.logical_graph().num_ops() == 1);
    REQUIRE(g.logical_graph().ops().front()->type() == OpType::CLEAR);

    auto& grad_again = g.get_or_create_grad(x, "x_grad");
    REQUIRE(&grad_again == &grad);
    REQUIRE(g.logical_graph().num_ops() == 1);

    auto node_text = x.to_string();
    REQUIRE(node_text.find("requires_grad=true") != std::string::npos);
    REQUIRE(node_text.find("grad='x_grad'") != std::string::npos);

    auto graph_text = g.to_string();
    REQUIRE(graph_text.find("NNGraph(name='grad'") != std::string::npos);
    REQUIRE(graph_text.find("Operations:") != std::string::npos);
}

TEST_CASE("NNGraph ToMermaidDelegatesToLogicalGraph", "[graph]")
{
    NNGraph nng("test_nn");

    auto& x = nng.tensor({2, 3}, "input", DataType::FP32);
    auto& w = nng.tensor({3, 4}, "weights", DataType::FP32);

    auto& y = nng.tensor({2, 4}, "output", DataType::FP32);

    // Add a simple operation
    nng.add_op(OpType::GEMM, GemmAttrs{}, {&x, &w}, {&y}, "matmul");

    // Test that NNGraph to_mermaid delegates to logical graph
    auto nn_mermaid = nng.to_mermaid();
    auto logical_mermaid = nng.logical_graph().to_mermaid();

    REQUIRE(nn_mermaid == logical_mermaid);

    // Check basic structure
    REQUIRE(nn_mermaid.find("graph TD") != std::string::npos);
    REQUIRE(nn_mermaid.find("classDef") != std::string::npos);
}

TEST_CASE("NNGraph MarkInputOutput", "[graph]")
{
    NNGraph g("test");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& w = g.tensor({3, 4}, "w", DataType::FP32);
    auto& y = g.tensor({2, 4}, "y", DataType::FP32);

    g.add_op(OpType::GEMM, GemmAttrs{}, {&x, &w}, {&y});

    x.mark_input(true);
    y.mark_output(true);

    REQUIRE(x.is_input());
    REQUIRE(y.is_output());
    REQUIRE(x.data().is_input());
    REQUIRE(y.data().is_output());
}

TEST_CASE("NNGraph Autograd Add Backward", "[graph]")
{
    // Example: z = add(alpha, x, beta, y) with z.backward()
    // Mimics PyTorch: z = alpha*x + beta*y, then z.backward()
    // Expected: grad_x = alpha, grad_y = beta (when grad_z = 1)
    NNGraph g("autograd_add");

    auto& x = g.tensor({2, 3}, "x", DataType::FP32);
    auto& y = g.tensor({2, 3}, "y", DataType::FP32);

    nntile::Scalar alpha = 2.0;
    nntile::Scalar beta = 3.0;

    auto& z = add(g, alpha, x, beta, y, "z");

    // Check grad_fn: z was produced by ADD op
    REQUIRE(z.grad_fn() != nullptr);
    REQUIRE(z.grad_fn()->type() == OpType::ADD);
    REQUIRE_FALSE(z.is_leaf());

    // x and y are leaves (inputs, no producer)
    REQUIRE(x.is_leaf());
    REQUIRE(y.is_leaf());
    REQUIRE(x.grad_fn() == nullptr);
    REQUIRE(y.grad_fn() == nullptr);

    // Build backward graph (PyTorch-style)
    z.backward();

    // After backward: x and y should have grad tensors
    REQUIRE(x.has_grad());
    REQUIRE(y.has_grad());

    // The backward graph adds: grad_x += alpha*grad_z, grad_y += beta*grad_z
    // With grad_z filled with 1.0: grad_x = alpha, grad_y = beta
    // Verify the graph structure - we have ADD, FILL, CLEAR, ADD_INPLACE ops
    const auto& ops = g.logical_graph().ops();
    REQUIRE(ops.size() >= 4);

    // Check that ADD_INPLACE ops exist for gradient accumulation
    size_t add_inplace_count = 0;
    for(const auto& op : ops)
    {
        if(op->type() == OpType::ADD_INPLACE)
        {
            ++add_inplace_count;
        }
    }
    REQUIRE(add_inplace_count == 2);

    // Verify grad tensor names exist
    REQUIRE(g.get_tensor("x_grad") != nullptr);
    REQUIRE(g.get_tensor("y_grad") != nullptr);
    REQUIRE(g.get_tensor("z_grad") != nullptr);
}
