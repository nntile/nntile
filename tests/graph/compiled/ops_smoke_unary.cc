#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GeluInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            gelu_inplace(x);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Gelutanh Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            gelutanh(x, "y");
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GelutanhInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            gelutanh_inplace(x);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph GelutanhBackward Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3}, "x", DataType::FP32);
            auto& dy = g.tensor({3}, "dy", DataType::FP32);
            auto& dx = g.tensor({3}, "dx", DataType::FP32);
            gelutanh_backward(x, dy, dx);
        },
        {"x", "dy", "dx"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ReluBackward Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3}, "x", DataType::FP32);
            auto& dy = g.tensor({3}, "dy", DataType::FP32);
            auto& dx = g.tensor({3}, "dx", DataType::FP32);
            relu_backward(x, dy, dx);
        },
        {"x", "dy", "dx"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Silu Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            silu(x, "y");
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SiluInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            silu_inplace(x);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SiluBackward Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3}, "x", DataType::FP32);
            auto& dy = g.tensor({3}, "dy", DataType::FP32);
            auto& dx = g.tensor({3}, "dx", DataType::FP32);
            silu_backward(x, dy, dx);
        },
        {"x", "dy", "dx"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SqrtInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            sqrt_inplace(x);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph AddInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4, 6}, "y", DataType::FP32);
            add_inplace(x, y, 1.0f, 0.5f);
        },
        {"x", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph HypotInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4, 6}, "y", DataType::FP32);
            hypot_inplace(x, y, 1.0f, 1.0f);
        },
        {"x", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph PowInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            pow_inplace(x, 1.0f, 2.0f);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph LogScalar Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4}, "x", DataType::FP32);
            log_scalar(x, "test_value");
        },
        {"x"}
    );
}
