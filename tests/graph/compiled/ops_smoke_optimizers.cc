#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SgdStep Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& grad = g.tensor({4}, "grad", DataType::FP32);
            auto& velocity = g.tensor({4}, "velocity", DataType::FP32);
            auto& p = g.tensor({4}, "p", DataType::FP32);
            sgd_step(grad, velocity, p, 1, 0.9f, 0.01f, 0.0f, 0.0f, false);
        },
        {"grad", "velocity", "p"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph AdamStep Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& grad = g.tensor({4}, "grad", DataType::FP32);
            auto& m = g.tensor({4}, "m", DataType::FP32);
            auto& v = g.tensor({4}, "v", DataType::FP32);
            auto& p = g.tensor({4}, "p", DataType::FP32);
            adam_step(grad, m, v, p, 1, 0.9f, 0.999f, 1e-8f, 0.01f, 0.0f);
        },
        {"grad", "m", "v", "p"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph AdamWStep Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& grad = g.tensor({4}, "grad", DataType::FP32);
            auto& m = g.tensor({4}, "m", DataType::FP32);
            auto& v = g.tensor({4}, "v", DataType::FP32);
            auto& p = g.tensor({4}, "p", DataType::FP32);
            adamw_step(grad, m, v, p, 1, 0.9f, 0.999f, 1e-8f, 0.01f, 0.0f);
        },
        {"grad", "m", "v", "p"}
    );
}
