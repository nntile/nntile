#include "compiled_test_utils.hh"

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Gather Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            gather(x, "y");
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Scatter Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            scatter(x, "y");
        },
        {"x"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph CopyIntersection Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& src = g.tensor({4}, "src", DataType::FP32);
            auto& dst = g.tensor({4}, "dst", DataType::FP32);
            copy_intersection(src, {0}, dst, {0});
        },
        {"src", "dst"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ScaleFiber Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4}, "x", DataType::FP32);
            auto& y = g.tensor({4, 6}, "y", DataType::FP32);
            scale_fiber(x, y, 1.0f, 0, 0);
        },
        {"x", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph ScaleSlice Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4}, "x", DataType::FP32);
            auto& y = g.tensor({4, 6}, "y", DataType::FP32);
            scale_slice(x, y, 1.0f, 1);
        },
        {"x", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Randn Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({4, 6}, "x", DataType::FP32);
            randn(x, 0, {4, 6}, 1234, 0.0f, 1.0f);
        },
        {}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Rope Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& sin = g.tensor({4, 6}, "sin", DataType::FP32);
            auto& cos = g.tensor({4, 6}, "cos", DataType::FP32);
            auto& src = g.tensor({4, 6}, "src", DataType::FP32);
            rope(sin, cos, src, "dst");
        },
        {"sin", "cos", "src"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph RopeBackward Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& sin = g.tensor({4, 6}, "sin", DataType::FP32);
            auto& cos = g.tensor({4, 6}, "cos", DataType::FP32);
            auto& dy = g.tensor({4, 6}, "dy", DataType::FP32);
            rope_backward(sin, cos, dy, "dx");
        },
        {"sin", "cos", "dy"}
    );
}
