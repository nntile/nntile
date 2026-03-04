/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/graph/nn.cc
 * Tests for NNGraph class.
 *
 * @version 1.1.0
 * */

// Include standard headers
#include <algorithm>
#include <string>

// Include third-party headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

// Include other NNTile headers
#include "context_fixture.hh"
#include "nntile/graph.hh"

using namespace nntile;
using namespace nntile::graph;
namespace gt = nntile::graph::tensor;

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph TensorNodeNullData", "[graph]")
{
    NNGraph g("test");
    REQUIRE_THROWS_AS(
        NNGraph::TensorNode(&g, nullptr, true),
        std::invalid_argument);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph TensorCreationAndLookup", "[graph]")
{
    NNGraph g("test");

    auto* x = g.tensor({2, 3}, "x", DataType::FP32, false);

    REQUIRE(x->name() == "x");
    REQUIRE_FALSE(x->requires_grad());
    REQUIRE(g.get_tensor("x") == x);
    REQUIRE(g.get_tensor("missing") == nullptr);
    REQUIRE(x->data() == g.tensor_graph().get_tensor_node("x"));

    auto names = g.tensor_names();
    REQUIRE(names.size() == 1);
    REQUIRE(std::find(names.begin(), names.end(), "x") != names.end());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph OpNullInputs", "[graph]")
{
    const Scalar gemm_alpha = GENERATE(Scalar(1.0));
    const bool trans_a = GENERATE(false);
    const bool trans_b = GENERATE(false);
    const Index ndim = GENERATE(Index(1));
    const Index batch_ndim = GENERATE(Index(0));

    NNGraph g("test");

    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    REQUIRE_THROWS_AS(gemm(nullptr, y, "out", gemm_alpha, trans_a, trans_b, ndim, batch_ndim),
                     std::invalid_argument);
    REQUIRE_THROWS_AS(gemm(x, nullptr, "out", gemm_alpha, trans_a, trans_b, ndim, batch_ndim),
                     std::invalid_argument);
    REQUIRE_THROWS_AS(gelu(static_cast<NNGraph::TensorNode*>(nullptr), "out"),
                     std::invalid_argument);
    REQUIRE_THROWS_AS(fill(Scalar(1.0), static_cast<NNGraph::TensorNode*>(nullptr)),
                     std::invalid_argument);
    REQUIRE_THROWS_AS(clear(static_cast<NNGraph::TensorNode*>(nullptr)),
                     std::invalid_argument);

    auto* z = gelu(x, "z");
    REQUIRE(z != nullptr);
    REQUIRE(g.num_ops() == 1);
    REQUIRE(g.tensor_graph().ops().front()->op_name() == "GELU");
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph GradHelpersAndToString", "[graph]")
{
    NNGraph g("grad");

    auto* x = g.tensor({4}, "x", DataType::FP32, false);
    REQUIRE_FALSE(g.requires_grad(x));

    g.set_requires_grad(x, true);
    REQUIRE(g.requires_grad(x));
    g.set_requires_grad(x, false);
    REQUIRE_FALSE(g.requires_grad(x));

    auto [grad, is_first] = g.get_or_create_grad(x, "x_grad");
    REQUIRE(x->has_grad());
    REQUIRE(grad == x->grad());
    REQUIRE(x->requires_grad());
    REQUIRE_FALSE(grad->requires_grad());
    REQUIRE_FALSE(g.requires_grad(grad));
    REQUIRE(is_first);
    REQUIRE(g.tensor_graph().num_ops() == 0);  // no CLEAR

    auto [grad_again, is_first_again] = g.get_or_create_grad(x, "x_grad");
    REQUIRE(grad_again == grad);
    REQUIRE_FALSE(is_first_again);
    REQUIRE(g.tensor_graph().num_ops() == 0);

    REQUIRE_THROWS_AS(g.get_or_create_grad(x, "different_grad_name"),
        std::invalid_argument);

    auto node_text = x->to_string();
    REQUIRE(node_text.find("requires_grad=true") != std::string::npos);
    REQUIRE(node_text.find("grad='x_grad'") != std::string::npos);

    auto graph_text = g.to_string();
    REQUIRE(graph_text.find("NNGraph(name='grad'") != std::string::npos);
    REQUIRE(graph_text.find("Operations:") != std::string::npos);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph ToMermaidDelegatesToTensorGraph", "[graph]")
{
    const Scalar gemm_alpha = GENERATE(Scalar(1.0));
    const bool trans_a = GENERATE(false);
    const bool trans_b = GENERATE(false);
    const Index ndim = GENERATE(Index(1));
    const Index batch_ndim = GENERATE(Index(0));

    NNGraph nng("test_nn");

    auto* x = nng.tensor({2, 3}, "input", DataType::FP32);
    auto* w = nng.tensor({3, 4}, "weights", DataType::FP32);

    auto* y = gemm(x, w, "output", gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

    // Test that NNGraph to_mermaid delegates to tensor graph
    auto nn_mermaid = nng.to_mermaid();
    auto tensor_mermaid = nng.tensor_graph().to_mermaid();

    REQUIRE(nn_mermaid == tensor_mermaid);

    // Check basic structure
    REQUIRE(nn_mermaid.find("graph TD") != std::string::npos);
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph MarkInputOutput", "[graph]")
{
    const Scalar gemm_alpha = GENERATE(Scalar(1.0));
    const bool trans_a = GENERATE(false);
    const bool trans_b = GENERATE(false);
    const Index ndim = GENERATE(Index(1));
    const Index batch_ndim = GENERATE(Index(0));

    NNGraph g("test");

    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* w = g.tensor({3, 4}, "w", DataType::FP32);
    auto* y = gemm(x, w, "y", gemm_alpha, trans_a, trans_b, ndim, batch_ndim);

    x->mark_input(true);
    y->mark_output(true);

    REQUIRE(x->is_input());
    REQUIRE(y->is_output());
    REQUIRE(x->data()->is_input());
    REQUIRE(y->data()->is_output());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add Backward", "[graph]")
{
    // Example: z = add(alpha, x, beta, y) with z.backward()
    // Mimics PyTorch: z = alpha*x + beta*y, then z.backward()
    // Expected: grad_x = alpha, grad_y = beta (when grad_z = 1)
    const Scalar alpha = GENERATE(Scalar(2.0));
    const Scalar beta = GENERATE(Scalar(3.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("autograd_add");

    auto* x = g.tensor({2, 3}, "x", DataType::FP32);
    auto* y = g.tensor({2, 3}, "y", DataType::FP32);

    auto* z = add(alpha, x, beta, y, "z");

    // z was produced by Add (NNGraph op)
    REQUIRE(z->has_producer());
    REQUIRE_FALSE(z->is_leaf());

    // x and y are leaves (inputs, no producer)
    REQUIRE(x->is_leaf());
    REQUIRE(y->is_leaf());
    REQUIRE_FALSE(x->has_producer());
    REQUIRE_FALSE(y->has_producer());

    // Set upstream gradient before backward (required)
    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());

    // Build backward graph (PyTorch-style)
    z->backward();

    // After backward: x and y should have grad tensors
    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());

    // The backward graph adds: grad_x += alpha*grad_z, grad_y += beta*grad_z
    // With grad_z filled with 1.0: grad_x = alpha, grad_y = beta
    // Verify the graph structure - we have ADD, FILL, CLEAR, ADD_INPLACE ops
    const auto& ops = g.tensor_graph().ops();
    REQUIRE(ops.size() >= 4);

    // Check that ADD_INPLACE ops exist for gradient accumulation
    size_t add_inplace_count = 0;
    for(const auto& op : ops)
    {
        if(op->op_name() == "ADD_INPLACE")
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

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add Chain", "[graph]")
{
    // Chain: w = x + y, z = w + u. Each tensor gets its gradient.
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("add_chain");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);
    auto* u = g.tensor({2, 2}, "u", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* z = add(add_alpha, w, add_beta, u, "z");

    REQUIRE(w->requires_grad());
    REQUIRE(w->has_producer());

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(u->has_grad());
    REQUIRE(w->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph Autograd Add Diamond", "[graph]")
{
    // Diamond: w = x + y, v = w + y, z = v + w.
    // w feeds into both v and z; backward must process v and z before w
    // so w.grad accumulates both contributions.
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("add_diamond");
    auto* x = g.tensor({2, 2}, "x", DataType::FP32);
    auto* y = g.tensor({2, 2}, "y", DataType::FP32);

    auto* w = add(add_alpha, x, add_beta, y, "w");
    auto* v = add(add_alpha, w, add_beta, y, "v");
    auto* z = add(add_alpha, v, add_beta, w, "z");

    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    z->backward();

    REQUIRE(x->has_grad());
    REQUIRE(y->has_grad());
    REQUIRE(w->has_grad());
    REQUIRE(v->has_grad());
}

TEST_CASE_METHOD(nntile::test::ContextFixture,
    "NNGraph BackwardRequiresGrad", "[graph]")
{
    // backward() must be called only when grad is already set
    const Scalar add_alpha = GENERATE(Scalar(1.0));
    const Scalar add_beta = GENERATE(Scalar(1.0));
    const Scalar grad_fill_val = GENERATE(Scalar(1.0));

    NNGraph g("backward_requires_grad");
    auto* x = g.tensor({2}, "x", DataType::FP32);
    auto* y = g.tensor({2}, "y", DataType::FP32);
    auto* z = add(add_alpha, x, add_beta, y, "z");

    REQUIRE_THROWS_AS(z->backward(), std::invalid_argument);

    // After setting grad, backward succeeds
    auto [z_grad, _] = g.get_or_create_grad(z, "z_grad");
    gt::fill(grad_fill_val, z_grad->data());
    REQUIRE_NOTHROW(z->backward());
}
