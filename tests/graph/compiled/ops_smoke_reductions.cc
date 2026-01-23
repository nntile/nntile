#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph LogSumExp Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({2, 3}, "x", DataType::FP32);
            auto& y = g.tensor({3}, "y", DataType::FP32);
            logsumexp(x, y, 0);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph MaxSumExp Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3}, "x", DataType::FP32);
            auto& y = g.tensor({2}, "y", DataType::FP32);
            maxsumexp(x, y, 0, 0);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SumProdFiber Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x1 = g.tensor({4, 6}, "x1", DataType::FP32);
            auto& x2 = g.tensor({4, 6}, "x2", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            sumprod_fiber(x1, x2, y, 0, 0, 1.0f, 0.0f);
        },
        {"x1", "x2"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph SumProdSlice Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x1 = g.tensor({4, 6}, "x1", DataType::FP32);
            auto& x2 = g.tensor({4, 6}, "x2", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            sumprod_slice(x1, x2, y, 1, 0, 1.0f, 0.0f);
        },
        {"x1", "x2"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph NormFiber Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            norm_fiber(x, y, 0, 0, 0, 1.0f, 0.0f);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph NormFiberInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            norm_fiber_inplace(x, y, 0, 0, 0, 1.0f, 0.0f);
        },
        {"x", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph NormSlice Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            norm_slice(x, y, 1, 0, 1.0f, 0.0f);
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph NormSliceInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            auto& y = g.tensor({4}, "y", DataType::FP32);
            norm_slice_inplace(x, y, 1, 0, 1.0f, 0.0f);
        },
        {"x", "y"}
    );
}
