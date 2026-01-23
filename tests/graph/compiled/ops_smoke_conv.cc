#include "compiled_test_utils.hh"

#include <array>

using namespace nntile;
using namespace nntile::graph;
using namespace nntile::graph::test;

namespace
{

constexpr std::array<Index, 2> kPadding = {0, 0};
constexpr std::array<Index, 2> kStride = {1, 1};
constexpr std::array<Index, 2> kDilation = {1, 1};

} // namespace

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Conv2dInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3, 3, 1, 1}, "x", DataType::FP32);
            auto& c = g.tensor({2, 2, 1, 1}, "c", DataType::FP32);
            auto& y = g.tensor({2, 2, 1, 1}, "y", DataType::FP32);
            conv2d_inplace(x, c, y, 1.0f, 0.0f, kPadding, kStride, kDilation);
        },
        {"x", "c", "y"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Conv2dBwdInputInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& dy = g.tensor({2, 2, 1, 1}, "dy", DataType::FP32);
            auto& c = g.tensor({2, 2, 1, 1}, "c", DataType::FP32);
            auto& dx = g.tensor({3, 3, 1, 1}, "dx", DataType::FP32);
            conv2d_bwd_input_inplace(dy, c, dx, 1.0f, 0.0f, kPadding, kStride, kDilation);
        },
        {"dy", "c", "dx"}
    );
}

TEST_CASE_METHOD(
    GraphTestFixture,
    "CompiledGraph Conv2dBwdWeightInplace Smoke",
    "[graph][compiled][smoke]")
{
    run_compiled_graph(
        [](LogicalGraph& g) {
            auto& x = g.tensor({3, 3, 1, 1}, "x", DataType::FP32);
            auto& dy = g.tensor({2, 2, 1, 1}, "dy", DataType::FP32);
            auto& dc = g.tensor({2, 2, 1, 1}, "dc", DataType::FP32);
            conv2d_bwd_weight_inplace(x, dy, dc, 1.0f, 0.0f, kPadding, kStride, kDilation);
        },
        {"x", "dy", "dc"}
    );
}
